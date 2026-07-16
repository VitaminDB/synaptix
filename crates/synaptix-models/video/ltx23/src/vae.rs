//! LTX-2.3 Video VAE decode: латент `[B,128,F',H',W']` → RGB `[B,3,F,H,W]`
//! (F=8·(F'−1)+1, H=32·H', W=32·W'). CausalVideoAutoencoder, decoder-путь.
//!
//! Конфиг чекпойнта: norm=pixel_norm, spatial_padding="zeros", causal_decoder=false,
//! timestep_conditioning=false → простой путь (без noise/timestep). Все res_x
//! resnet'ы in==out (shortcut/norm3 = Identity).

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_ops::conv::conv3d::conv3d;

use crate::config::VaeConfig;
use crate::loader::{LtxCheckpoint, VAE_PREFIX};
use crate::LtxError;

type R<T> = Result<T, SynaptixError>;
const PIXEL_EPS: f64 = 1e-8;
const CPU_STITCH_BYTES: usize = 2 << 30;

/// Жадный подбор сетки тайлов: дробим длинную ось, пока haloed-площадь тайла
/// не влезет в cap (halo прибавляется только к тайлящейся оси — при n=1 это
/// полный кадр без halo, иначе малые разрешения over-тайлятся).
fn pick_grid(hp: usize, wp: usize, halo: usize, haloed_cap: usize) -> (usize, usize) {
    let mut nh = 1usize;
    let mut nw = 1usize;
    loop {
        let ch = hp.div_ceil(nh);
        let cw = wp.div_ceil(nw);
        let eh = ch + if nh > 1 { 2 * halo } else { 0 };
        let ew = cw + if nw > 1 { 2 * halo } else { 0 };
        if eh * ew <= haloed_cap || (nh >= hp && nw >= wp) {
            break;
        }
        if ch >= cw {
            nh += 1;
        } else {
            nw += 1;
        }
    }
    (nh.min(hp), nw.min(wp))
}

/// PixelNorm: `x / sqrt(mean(x², dim=1) + eps)` (per-location RMS по каналам).
/// Считается в f32. `silu=true` фьюзит активацию в то же ядро. Fused-путь
/// (CUDA) — один launch; CPU/неподдержка → decomposed.
fn pixel_norm_act(x: &Tensor, silu: bool) -> R<Tensor> {
    match x.pixel_norm_fused(PIXEL_EPS as f32, silu) {
        Ok(t) => return Ok(t),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let ms = xf.sqr()?.mean_keepdim(1)?; // [B,1,D,H,W]
    let denom = ms.add_scalar(PIXEL_EPS as f32)?.sqrt()?;
    let y = xf.broadcast_div(&denom)?.to_dtype(dt)?;
    if silu { y.silu() } else { Ok(y) }
}

/// CausalConv3d 3×3×3 stride1, spatial zero-pad 1. `causal=false` (decoder):
/// temporal replicate-pad первого+последнего кадра по (k−1)/2=1 (симметрично).
/// `causal=true` (encoder): front-pad (k−1)=2 кадра (replicate первого), без
/// заднего паддинга — будущие кадры не влияют на прошлые.
fn vae_conv_c(x: &Tensor, w: &Tensor, b: &Tensor, causal: bool) -> R<Tensor> {
    let x = x.contiguous()?;
    let d = x.dims()[2];
    let first = x.narrow(2, 0, 1)?.contiguous()?;
    let xp = if causal {
        // front-pad 2× первого кадра
        Tensor::cat(&[&first, &first, &x], 2)?.contiguous()?
    } else {
        let last = x.narrow(2, d - 1, 1)?.contiguous()?;
        Tensor::cat(&[&first, &x, &last], 2)?.contiguous()?
    };
    conv3d(&xp, w, Some(b), (1, 1, 1), (0, 1, 1), (1, 1, 1))
}

/// depth-to-space: `[B, c·sd·sh·sw, D, H, W] → [B, c, D·sd, H·sh, W·sw]`.
/// einops `b (c p1 p2 p3) d h w -> b c (d p1)(h p2)(w p3)` (c внешний).
fn depth_to_space(x: &Tensor, sd: usize, sh: usize, sw: usize) -> R<Tensor> {
    let (b, cc, d, h, w) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3], x.dims()[4]);
    let c = cc / (sd * sh * sw);
    x.reshape(vec![b, c, sd, sh, sw, d, h, w])?
        .permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?
        .contiguous()?
        .reshape(vec![b, c, d * sd, h * sh, w * sw])
}

/// unpatchify `b (c p r q) f h w -> b c (f p)(h q)(w r)`, p=1, q=r=patch.
fn unpatchify(x: &Tensor, patch: usize) -> R<Tensor> {
    let (b, cc, f, h, w) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3], x.dims()[4]);
    let c = cc / (patch * patch);
    x.reshape(vec![b, c, 1, patch, patch, f, h, w])?
        .permute(vec![0, 1, 5, 2, 6, 4, 7, 3])?
        .contiguous()?
        .reshape(vec![b, c, f, h * patch, w * patch])
}

/// compute-dtype VAE-декода: BF16 (×2 тайл, ×2+ conv-скорость; как офиц.
/// bf16-пайплайн).
pub(crate) fn vae_decode_dtype() -> DType {
    DType::BF16
}

struct Conv {
    w: Tensor,
    b: Tensor,
    causal: bool,
}
impl Conv {
    fn load(ckpt: &LtxCheckpoint, prefix: &str, device: Device, dt: DType) -> Result<Self, LtxError> {
        Self::load_c(ckpt, prefix, device, false, dt)
    }
    /// `causal=true` → encoder-путь (front-pad).
    fn load_c(ckpt: &LtxCheckpoint, prefix: &str, device: Device, causal: bool, dt: DType) -> Result<Self, LtxError> {
        Ok(Self {
            w: ckpt.get(&format!("{prefix}.conv.weight"))?.to_device(device)?.to_dtype(dt)?,
            b: ckpt.get(&format!("{prefix}.conv.bias"))?.to_device(device)?.to_dtype(dt)?,
            causal,
        })
    }
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        vae_conv_c(x, &self.w, &self.b, self.causal)
    }
}

/// patchify: `b c (h q)(w r) -> b (c r q) h w` (5D, patch_t=1): spatial 4×4-патч в
/// каналы (порядок каналов c,r,q). `[B,C,F,H,W] → [B,C·patch²,F,H/patch,W/patch]`.
fn patchify(x: &Tensor, patch: usize) -> R<Tensor> {
    let (b, c, f, h, w) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3], x.dims()[4]);
    let (hp, wp) = (h / patch, w / patch);
    // [b,c,f,hp,q,wp,r] → канал-порядок (c, r, q) → [b, c·r·q, f, hp, wp]
    x.reshape(vec![b, c, f, hp, patch, wp, patch])?
        .permute(vec![0, 1, 6, 4, 2, 3, 5])? // b,c,r,q,f,hp,wp
        .contiguous()?
        .reshape(vec![b, c * patch * patch, f, hp, wp])
}

/// space_to_depth: `b c (d p1)(h p2)(w p3) -> b (c p1 p2 p3) d h w` (порядок
/// каналов c,p1,p2,p3). Downsample по (sd,sh,sw) с переносом в каналы.
fn space_to_depth(x: &Tensor, sd: usize, sh: usize, sw: usize) -> R<Tensor> {
    let (b, c, dd, hh, ww) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3], x.dims()[4]);
    let (d, h, w) = (dd / sd, hh / sh, ww / sw);
    x.reshape(vec![b, c, d, sd, h, sh, w, sw])?
        .permute(vec![0, 1, 3, 5, 7, 2, 4, 6])? // b,c,p1,p2,p3,d,h,w
        .contiguous()?
        .reshape(vec![b, c * sd * sh * sw, d, h, w])
}

/// ResnetBlock3D (in==out): PixelNorm→silu→conv1→PixelNorm→silu→conv2 + residual.
struct Resnet {
    conv1: Conv,
    conv2: Conv,
}
impl Resnet {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let h = self.conv1.fwd(&pixel_norm_act(x, true)?)?;
        let h = self.conv2.fwd(&pixel_norm_act(&h, true)?)?;
        x.add(&h)
    }
}

enum UpBlock {
    Res(Vec<Resnet>),
    Upsample { conv: Conv, sd: usize, sh: usize, sw: usize },
}

fn blk_kind(b: &UpBlock) -> &'static str {
    match b {
        UpBlock::Res(_) => "res",
        UpBlock::Upsample { .. } => "upsample",
    }
}

pub struct VaeDecoder {
    mean: Tensor, // [1,128,1,1,1]
    std: Tensor,
    conv_in: Conv,
    up_blocks: Vec<UpBlock>,
    conv_out: Conv,
    patch_size: usize,
    device: Device,
    dt: DType,
}

impl VaeDecoder {
    pub fn load(ckpt: &LtxCheckpoint, device: Device) -> Result<Self, LtxError> {
        let dt = vae_decode_dtype();
        let cfg: &VaeConfig = &ckpt.config.vae;
        let dec = format!("{VAE_PREFIX}.decoder");
        let stat = |n: &str| -> Result<Tensor, LtxError> {
            Ok(ckpt.get(&format!("{VAE_PREFIX}.per_channel_statistics.{n}"))?
                .to_device(device)?.to_dtype(dt)?.reshape(vec![1, 128, 1, 1, 1])?)
        };
        // декодер строится по reversed(decoder_blocks)
        let mut up_blocks = Vec::new();
        let mut bi = 0usize;
        for blk in cfg.decoder_blocks.iter().rev() {
            let name = &blk.0;
            let p = format!("{dec}.up_blocks.{bi}");
            if name == "res_x" {
                let n = blk.1.num_layers.unwrap_or(1);
                let mut res = Vec::with_capacity(n);
                for j in 0..n {
                    res.push(Resnet {
                        conv1: Conv::load(ckpt, &format!("{p}.res_blocks.{j}.conv1"), device, dt)?,
                        conv2: Conv::load(ckpt, &format!("{p}.res_blocks.{j}.conv2"), device, dt)?,
                    });
                }
                up_blocks.push(UpBlock::Res(res));
            } else {
                let (sd, sh, sw) = match name.as_str() {
                    "compress_all" => (2, 2, 2),
                    "compress_time" => (2, 1, 1),
                    "compress_space" => (1, 2, 2),
                    other => return Err(LtxError::Config(format!("unknown decoder block: {other}"))),
                };
                up_blocks.push(UpBlock::Upsample {
                    conv: Conv::load(ckpt, &format!("{p}.conv"), device, dt)?,
                    sd, sh, sw,
                });
            }
            bi += 1;
        }
        Ok(Self {
            mean: stat("mean-of-means")?,
            std: stat("std-of-means")?,
            conv_in: Conv::load(ckpt, &format!("{dec}.conv_in"), device, dt)?,
            up_blocks,
            conv_out: Conv::load(ckpt, &format!("{dec}.conv_out"), device, dt)?,
            patch_size: cfg.patch_size,
            device,
            dt,
        })
    }

    /// Декод латента `[B,128,F',H',W']` → RGB `[B,3,F,H,W]`. Двухуровневый
    /// авто-тайлинг: ВРЕМЕННЫЕ чанки (halo ±8 латент-кадров = RF декодера,
    /// маппинг кадров 8·(f−1)+1 выравнивается кропом) × ПРОСТРАНСТВЕННАЯ сетка
    /// внутри чанка. Память окна ∝ chunk, НЕ длительности — иначе длинные видео
    /// (20s+ = 481 кадр) не влезают ни в какую карту, а чисто пространственное
    /// мельчение вырождается (halo 6 доминирует над ядром тайла).
    /// Размер чанка авто: минимум суммарной работы (t-оверхед × sp-оверхед) при
    /// пер-окно ≤ бюджет.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, LtxError> {
        let (hp, wp) = (latent.dims()[3], latent.dims()[4]);
        let fp = latent.dims()[2];
        let halo: usize = 6;
        // RF декодера по времени: расчёт Σ conv-стадий ≈ ±12.5 латента, но
        // эмпирика гейта (512×288, fp=37, tc=4): на дистанции РОВНО 13 от края
        // окна ещё ±2 кванта u8 на полулатенте, ≥14 чисто → RF ≈ 13.5. Halo 16
        // с запасом — бит-в-бит к полному декоду (md5 PPM 289/289).
        let t_halo: usize = 16;
        let budget = self.vram_budget();
        let k_bytes = self.k_bytes();
        // Полный объём влезает одним окном → прямой декод без временных чанков:
        // любой чанкинг платит halo-пересчётом (5s HD blend-путь = ×1.44 латентов,
        // 20.2s vs 9.25s одним окном) и швами. Условие = nh=nw=1 в decode_spatial.
        let f_out_full = 8 * fp.saturating_sub(1) + 1;
        let fits_single = hp * wp * f_out_full * k_bytes <= budget;
        // Temporal-BLEND (дефолт, как офиц. tiled_decode: overlap 24 кадра +
        // линейный кроссфейд): окна с мини-halo 3-4 латента вместо RF-16 →
        // full-frame окна влезают, halo-оверхед ~1.5× вместо ~10×.
        let t_blend: usize = 3;
        if t_blend > 0 && !fits_single && fp > 2 * t_blend + 8 && latent.device().is_cuda() {
            return self.decode_tblend(latent, budget, t_blend);
        }
        // авто: перебор кандидатов, cost = Σ(окна чанков × haloed-площадь
        // тайлов) — полный объём прогоняемых через декодер cell-кадров.
        let t_core: usize = {
            let mut best = (fp, usize::MAX);
            for &tc in &[fp, 32, 24, 16, 12, 8, 6, 4] {
                if tc > fp || tc == 0 {
                    continue;
                }
                let len = (tc + 2 * t_halo).min(fp);
                let f_w = 8 * (len - 1) + 1;
                let cap = (budget / (k_bytes * f_w)).max(16);
                let (nh, nw) = pick_grid(hp, wp, halo, cap);
                let eh = hp.div_ceil(nh) + if nh > 1 { 2 * halo } else { 0 };
                let ew = wp.div_ceil(nw) + if nw > 1 { 2 * halo } else { 0 };
                let cost = fp.div_ceil(tc) * f_w * nh * nw * eh * ew;
                if cost < best.1 {
                    best = (tc, cost);
                }
            }
            best.0
        };
        let t_win = (t_core + 2 * t_halo).min(fp);
        if t_core >= fp || t_win >= fp {
            return self.decode_spatial(latent, budget);
        }
        let prof = crate::runtime::ltx_vae_prof();
        // чанки ВСЕГДА на CPU: GPU-чанк (десятки MB) живёт внутри большого
        // пул-блока и не даёт trim'у вернуть блок драйверу — к 3-4-му чанку
        // транзиенты декода (~2GB/чанк) съедают VRAM. D2H чанка копеечный.
        let to_cpu = latent.device().is_cuda();
        let mut parts: Vec<Tensor> = Vec::new();
        let mut s = 0usize;
        while s < fp {
            let e = (s + t_core).min(fp);
            // окно ФИКСИРОВАННОЙ длины t_win: у краёв сдвигается внутрь массива
            // (излишний halo безвреден) — один shape во всех чанках; разные
            // shapes удерживали VRAM по ~размеру транзиентов на каждый чанк
            // (single-прогон с одним shape не рос).
            let ws = s.saturating_sub(t_halo).min(fp - t_win);
            let we = ws + t_win;
            let live_gb = || synaptix_core::memory::cuda_pool::cuda_allocated_bytes() as f64 / 1e9;
            if prof { eprintln!("[VAE_TCHUNK] pre-decode live={:.1}GB", live_gb()); }
            let sub = latent.narrow(2, ws, we - ws)?.contiguous()?;
            let rgb_w = self.decode_spatial(&sub, budget)?;
            if prof { eprintln!("[VAE_TCHUNK] post-decode live={:.1}GB", live_gb()); }
            // ядро в локальных кадрах окна: латент j=0 → кадр [0,1), j≥1 →
            // [8(j−1)+1, 8j+1); ws>0 гарантирует js≥1 (t_halo≥1) → кроп выровнен
            // с глобальной развёрткой 8·(f−1)+1.
            let js = s - ws;
            let fls = if js == 0 { 0 } else { 8 * (js - 1) + 1 };
            let flen = if s == 0 { 8 * (e - 1) + 1 } else { 8 * (e - s) };
            let mut part = rgb_w.narrow(2, fls, flen)?.contiguous()?;
            if to_cpu {
                part = part.to_device(Device::Cpu)?;
            }
            drop(rgb_w);
            // sync ВСЕХ стримов (default+alloc+loader): транзиенты чанка
            // освобождаются cuMemFreeAsync в очередях СВОИХ стримов — sync
            // только default оставлял frees alloc_stream'а pending, пул не
            // пополнялся и каждый чанк ел свежую VRAM (−3.9GB/чанк, OOM на 4-м
            // при live=0 по учёту).
            if let Device::Cuda(ord) = self.device {
                let _ = synaptix_core::device::cuda::synchronize_all(ord);
            }
            if prof {
                let free = if let Device::Cuda(ord) = self.device {
                    synaptix_core::device::cuda::mem_info(ord).map(|(f, _)| f).unwrap_or(0)
                } else { 0 };
                eprintln!("[VAE_TCHUNK] лат[{s}..{e}) окно[{ws}..{we}) кадры {fls}+{flen}{} free={:.1}GB live={:.1}GB",
                    if to_cpu { " (cpu)" } else { "" }, free as f64 / 1e9,
                    synaptix_core::memory::cuda_pool::cuda_allocated_bytes() as f64 / 1e9);
                for (bytes, count) in synaptix_core::memory::cuda_pool::live_alloc_top(8) {
                    eprintln!("[ALLOC_TOP] {:>12} B × {count} = {:.2}GB", bytes, bytes as f64 * count as f64 / 1e9);
                }
            }
            parts.push(part);
            s = e;
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Ok(Tensor::cat(&refs, 2)?)
    }

    /// Временно́й blend-декод: чанки латента с мини-halo (слева t_blend+1 — фаза
    /// развёртки 8·(f−1)+1, справа t_blend), выход соседних чанков перекрывается
    /// на 2·8·t_blend кадров и сшивается линейным кроссфейдом (как офиц.
    /// tiled_decode). Окна по возможности full-frame → spatial-halo не платится.
    /// Маппинг лок→глоб кадров: ws=0 → p↔p; ws>0 → p≥1 ↔ 8·ws+p (p=0 дроп).
    fn decode_tblend(&self, latent: &Tensor, budget: usize, t_blend: usize) -> Result<Tensor, LtxError> {
        let (hp, wp) = (latent.dims()[3], latent.dims()[4]);
        let fp = latent.dims()[2];
        let prof = crate::runtime::ltx_vae_prof();
        let cells = hp * wp;
        // авто-ядро чанка: окно (tc + 2·t_blend + 1) full-frame в бюджет
        // ядро ≥ 2·t_blend+1: иначе вклад чанка короче двух рамп (head 2·bz +
        // tail 2·bz) → usize-wrap в narrow (наблюдалось: alloc 2^61 на tc=4).
        let tc_min = 2 * t_blend + 1;
        // max длина окна из бюджета: cells·(8·(len−1)+1)·k ≤ budget
        let t_core: usize = {
            let cap_frames = (budget / (self.k_bytes() * cells)).max(1);
            let len_max = ((cap_frames.saturating_sub(1)) / 8 + 1).max(1);
            len_max.saturating_sub(2 * t_blend + 1).clamp(tc_min, 32)
        };
        let bz = 8 * t_blend; // кадров рампы на сторону
        if prof {
            eprintln!("[VAE_TBLEND] fp={fp} t_core={t_core} blend={bz}к окна full-frame {cells} cells");
        }
        // веса рампы (вверх) на зоне 2·bz кадров: w_up[j]=(j+0.5)/(2·bz).
        // Рампы и кроссфейд на GPU в f32 (мелкие ядра); средние части D2H в
        // родном dt — прежний путь сливал ВЕСЬ выход на CPU в f32 (каст 5GB+ на
        // CPU и двойной D2H-трафик).
        let mk_w = |up: bool, n: usize| -> Result<Tensor, LtxError> {
            let v: Vec<f32> = (0..n)
                .map(|j| {
                    let w = (j as f32 + 0.5) / (n as f32);
                    if up { w } else { 1.0 - w }
                })
                .collect();
            Ok(Tensor::from_vec(v, vec![1, 1, n, 1, 1], Device::Cpu)?.to_device(self.device)?)
        };
        let z = 2 * bz;
        let (w_dn, w_up) = (mk_w(false, z)?, mk_w(true, z)?);
        let mut out_parts: Vec<Tensor> = Vec::new();
        let mut pending_tail: Option<Tensor> = None; // сырые последние 2·bz кадров пред. чанка (gpu, dt)
        let mut s = 0usize;
        while s < fp {
            // хвост короче t_blend+2 поглощается текущим чанком: иначе окно
            // последнего не покроет его левую рампу (we клампнется в fp)
            let e = if fp - (s + t_core).min(fp) < t_blend + 2 { fp } else { s + t_core };
            let ws = s.saturating_sub(t_blend + 1);
            let we = (e + t_blend).min(fp);
            let sub = latent.narrow(2, ws, we - ws)?.contiguous()?;
            let rgb_w = self.decode_spatial(&sub, budget)?;
            // вклад чанка в глоб. кадрах: [ext0, ext1)
            let g0 = if s == 0 { 0 } else { 8 * (s - 1) + 1 };
            let g1 = 8 * (e - 1) + 1;
            let ext0 = if s == 0 { 0 } else { g0 - bz };
            let ext1 = if e == fp { g1 } else { g1 + bz };
            let (pl0, pl1) = if ws == 0 {
                (ext0, ext1)
            } else {
                (ext0 - 8 * ws, ext1 - 8 * ws)
            };
            // decode_spatial при cpu-stitch уже вернул CPU-тензор — стык тогда
            // блендится на CPU (редкий путь: окно одновременно >2GB и тайловое)
            let contrib = rgb_w.narrow(2, pl0, pl1 - pl0)?.contiguous()?;
            drop(rgb_w);
            if prof {
                eprintln!("[VAE_TBLEND] лат[{s}..{e}) окно[{ws}..{we}) глоб[{ext0}..{ext1})");
            }
            let n = pl1 - pl0;
            let mut off = 0usize;
            if let Some(tail) = pending_tail.take() {
                // кроссфейд 2·bz кадров в f32: tail вниз, голова вклада вверх
                let head = contrib.narrow(2, 0, z)?.to_dtype(DType::F32)?;
                let (wd, wu) = if head.device().is_cuda() {
                    (w_dn.clone(), w_up.clone())
                } else {
                    (w_dn.to_device(Device::Cpu)?, w_up.to_device(Device::Cpu)?)
                };
                let blended = tail
                    .to_dtype(DType::F32)?
                    .broadcast_mul(&wd)?
                    .add(&head.broadcast_mul(&wu)?)?
                    .to_dtype(self.dt)?
                    .to_device(Device::Cpu)?;
                out_parts.push(blended);
                off = z;
            }
            let tail_z = if e == fp { 0 } else { 2 * bz };
            out_parts.push(
                contrib.narrow(2, off, n - off - tail_z)?.contiguous()?.to_device(Device::Cpu)?,
            );
            if tail_z > 0 {
                pending_tail = Some(contrib.narrow(2, n - tail_z, tail_z)?.contiguous()?);
            }
            drop(contrib);
            if let Device::Cuda(ord) = self.device {
                let _ = synaptix_core::device::cuda::synchronize_all(ord);
            }
            s = e;
        }
        let refs: Vec<&Tensor> = out_parts.iter().collect();
        Ok(Tensor::cat(&refs, 2)?)
    }

    /// Бюджет пика VRAM на окно декода: фактически свободная VRAM − резерв 3GB
    /// (cat-окна, латент, фрагментация). Перед замером trim пула: после denoise
    /// кэш аллокатора держит гигабайты освобождённых блоков — mem_info без трима
    /// врёт в меньшую сторону. Прежняя константа 18GB карту не спрашивала:
    /// мелкие GPU падали.
    fn vram_budget(&self) -> usize {
        if let Device::Cuda(ord) = self.device {
            let _ = synaptix_core::device::cuda::synchronize_all(ord);
            let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(ord);
            match synaptix_core::device::cuda::mem_info(ord) {
                Ok((free, _)) => {
                    let b = free.saturating_sub(3 << 30).max(4 << 30);
                    if crate::runtime::ltx_vae_prof() {
                        eprintln!(
                            "[VAE_BUDGET] free={:.2}GB budget={:.2}GB live={:.2}GB",
                            free as f64 / 1e9, b as f64 / 1e9,
                            synaptix_core::memory::cuda_pool::cuda_allocated_bytes() as f64 / 1e9
                        );
                    }
                    b
                }
                Err(_) => 18_000_000_000,
            }
        } else {
            18_000_000_000
        }
    }

    /// байт/(латент-площадь·кадр): эмпирика F32 (FullHD-тайл haloed 21×22@121к →
    /// 12.9GB ⇒ K ≈ 231000); BF16 вдвое меньше. Уточнено по факту: 20s-HD-тайл
    /// 16×17@481к bf16 НЕ влез в 21.5GB ⇒ K_bf16 ≥ 164000 — слагаемое сверх
    /// линейного по F; берём 175000/350000 с запасом.
    fn k_bytes(&self) -> usize {
        if self.dt == DType::F32 { 350_000 } else { 175_000 }
    }

    /// Пространственный уровень: полный кадр либо сетка тайлов под бюджет.
    fn decode_spatial(&self, latent: &Tensor, budget: usize) -> Result<Tensor, LtxError> {
        let (hp, wp) = (latent.dims()[3], latent.dims()[4]);
        let fp = latent.dims()[2];
        let halo: usize = 6;
        let f_out = 8 * (fp.saturating_sub(1)) + 1;
        let (nh, nw) = match crate::runtime::vae_grid() {
            Some((a, b)) => (a.max(1), b.max(1)),
            None => {
                let haloed_cap = (budget / (self.k_bytes() * f_out.max(1))).max(16);
                pick_grid(hp, wp, halo, haloed_cap)
            }
        };
        if nh <= 1 && nw <= 1 {
            return self.decode_window(latent);
        }
        self.decode_spatial_tiled(latent, nh, nw, halo)
    }

    /// Пространственный тайлинг: сетка `nh×nw` ядер по латент-H/W, halo вокруг
    /// каждого, декод окна, обрезка до ядра (×32), сшивка cat по W затем по H.
    /// Крупный выход (>CPU_STITCH_BYTES, например 20s HD ≈ 2.6GB bf16) сшивается
    /// на CPU: каждый crop сразу D2H, GPU держит только пер-тайловые активации —
    /// иначе выход+cat-транзиенты добавляют ~3× размера выхода к пику VRAM.
    /// Выход decode терминальный (PNG/mp4 через host), даунстрим CPU-совместим.
    fn decode_spatial_tiled(&self, latent: &Tensor, nh: usize, nw: usize, halo: usize) -> Result<Tensor, LtxError> {
        let prof = crate::runtime::ltx_vae_prof();
        let (hp, wp) = (latent.dims()[3], latent.dims()[4]);
        let f_out = 8 * (latent.dims()[2].saturating_sub(1)) + 1;
        let out_bytes = self.dt.bytes_for_numel(3 * f_out * (hp * 32) * (wp * 32));
        let cpu_stitch = out_bytes > CPU_STITCH_BYTES
            && latent.device().is_cuda();
        // границы ядер по осям
        let bounds = |n: usize, sz: usize| -> Vec<(usize, usize)> {
            (0..n).map(|i| {
                let s = i * sz / n;
                let e = (i + 1) * sz / n;
                (s, e - s)
            }).collect()
        };
        let hb = bounds(nh, hp);
        let wb = bounds(nw, wp);
        let mut rows: Vec<Tensor> = Vec::with_capacity(nh);
        for &(h0, lh) in &hb {
            let hs = h0.saturating_sub(halo);
            let he = (h0 + lh + halo).min(hp);
            let mut cols: Vec<Tensor> = Vec::with_capacity(nw);
            for &(w0, lw) in &wb {
                let ws = w0.saturating_sub(halo);
                let we = (w0 + lw + halo).min(wp);
                let sub = latent.narrow(3, hs, he - hs)?.narrow(4, ws, we - ws)?.contiguous()?;
                let rgb = self.decode_window(&sub)?; // [1,3,F,(he-hs)·32,(we-ws)·32]
                // обрезка до ядра: смещение = (h0-hs)·32, (w0-ws)·32; размер lh·32×lw·32
                let mut crop = rgb
                    .narrow(3, (h0 - hs) * 32, lh * 32)?
                    .narrow(4, (w0 - ws) * 32, lw * 32)?
                    .contiguous()?;
                if cpu_stitch {
                    crop = crop.to_device(Device::Cpu)?;
                }
                if prof {
                    eprintln!("[VAE_TILE] H[{h0}+{lh}] W[{w0}+{lw}] окно {:?} → ядро {:?}{}",
                        rgb.dims(), crop.dims(), if cpu_stitch { " (cpu-stitch)" } else { "" });
                }
                cols.push(crop);
            }
            let crefs: Vec<&Tensor> = cols.iter().collect();
            rows.push(Tensor::cat(&crefs, 4)?); // сшивка по W
        }
        let rrefs: Vec<&Tensor> = rows.iter().collect();
        Ok(Tensor::cat(&rrefs, 3)?) // сшивка по H
    }

    /// Декод одного латент-окна `[B,128,F',H',W']` → RGB `[B,3,F,H,W]` (весь
    /// декодер; пик VRAM ∝ F'·H'·W').
    fn decode_window(&self, latent: &Tensor) -> Result<Tensor, LtxError> {
        let prof = crate::runtime::ltx_vae_prof();
        let ord = if let Device::Cuda(o) = self.device { o } else { 0 };
        let tick = |label: &str, t: std::time::Instant| {
            if prof {
                let _ = synaptix_core::device::cuda::synchronize(ord);
                eprintln!("[VAE_PROF] {label}: {:.3}s live={:.2}GB",
                    t.elapsed().as_secs_f32(),
                    synaptix_core::memory::cuda_pool::cuda_allocated_bytes() as f64 / 1e9);
            }
        };
        let x = latent.to_device(self.device)?.to_dtype(self.dt)?;
        // un_normalize: x*std + mean
        let mut x = x.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let t0 = std::time::Instant::now();
        x = self.conv_in.fwd(&x)?;
        tick("conv_in", t0);
        for (bi, blk) in self.up_blocks.iter().enumerate() {
            let tb = std::time::Instant::now();
            x = match blk {
                UpBlock::Res(res) => {
                    let mut h = x;
                    for r in res {
                        h = r.forward(&h)?;
                    }
                    h
                }
                UpBlock::Upsample { conv, sd, sh, sw } => {
                    let c = conv.fwd(&x)?;
                    let mut up = depth_to_space(&c, *sd, *sh, *sw)?;
                    if *sd == 2 {
                        // drop first temporal frame после ×2 по времени
                        let d = up.dims()[2];
                        up = up.narrow(2, 1, d - 1)?.contiguous()?;
                    }
                    up
                }
            };
            if prof {
                let _ = synaptix_core::device::cuda::synchronize(ord);
                eprintln!("[VAE_PROF] up_block[{bi}] {:?} → {:?}: {:.3}s live={:.2}GB",
                    blk_kind(blk), x.dims(), tb.elapsed().as_secs_f32(),
                    synaptix_core::memory::cuda_pool::cuda_allocated_bytes() as f64 / 1e9);
            }
        }
        let t1 = std::time::Instant::now();
        let x = pixel_norm_act(&x, true)?;
        let x = self.conv_out.fwd(&x)?;
        let x = unpatchify(&x, self.patch_size)?;
        tick("conv_out+unpatchify", t1);
        Ok(x)
    }
}

/// Блок энкодера: стек resnet'ов (res_x) или SpaceToDepthDownsample (compress_*).
enum DownBlock {
    Res(Vec<Resnet>),
    /// SpaceToDepthDownsample: causal-pad (если sd=2) → s2d-skip (group-mean) +
    /// causal-conv(out/prod) → s2d → сумма. `out`=выходные каналы, `g`=group_size.
    Down { conv: Conv, sd: usize, sh: usize, sw: usize, out: usize, g: usize },
}

/// VAE encoder: кадры `[B,3,F,H,W]` (в [−1,1]) → нормализованный латент-means
/// `[B,128,F',H',W']` (F'=1+(F−1)/8, H'=H/32, W'=W/32). Зеркало декодера, convs
/// causal=True. UNIFORM logvar → берём первые 128 каналов (means), нормализуем.
pub struct VaeEncoder {
    mean: Tensor, // [1,128,1,1,1]
    std: Tensor,
    conv_in: Conv,
    down_blocks: Vec<DownBlock>,
    conv_out: Conv,
    patch_size: usize,
    device: Device,
}

impl VaeEncoder {
    pub fn load(ckpt: &LtxCheckpoint, device: Device) -> Result<Self, LtxError> {
        let cfg: &VaeConfig = &ckpt.config.vae;
        let enc = format!("{VAE_PREFIX}.encoder");
        let stat = |n: &str| -> Result<Tensor, LtxError> {
            Ok(ckpt.get(&format!("{VAE_PREFIX}.per_channel_statistics.{n}"))?
                .to_device(device)?.to_dtype(DType::F32)?.reshape(vec![1, 128, 1, 1, 1])?)
        };
        // вход после patchify: 3·patch² каналов → conv_in → 128
        let mut ch = 128usize;
        let mut down_blocks = Vec::new();
        for (bi, blk) in cfg.encoder_blocks.iter().enumerate() {
            let name = blk.0.as_str();
            let p = format!("{enc}.down_blocks.{bi}");
            if name == "res_x" {
                let n = blk.1.num_layers.unwrap_or(1);
                let mut res = Vec::with_capacity(n);
                for j in 0..n {
                    res.push(Resnet {
                        conv1: Conv::load_c(ckpt, &format!("{p}.res_blocks.{j}.conv1"), device, true, DType::F32)?,
                        conv2: Conv::load_c(ckpt, &format!("{p}.res_blocks.{j}.conv2"), device, true, DType::F32)?,
                    });
                }
                down_blocks.push(DownBlock::Res(res));
            } else {
                let (sd, sh, sw) = match name {
                    "compress_all_res" => (2, 2, 2),
                    "compress_time_res" => (2, 1, 1),
                    "compress_space_res" => (1, 2, 2),
                    other => return Err(LtxError::Config(format!("unknown encoder block: {other}"))),
                };
                let mult = blk.1.multiplier.unwrap_or(2.0) as usize;
                let out = ch * mult;
                let prod = sd * sh * sw;
                let g = ch * prod / out; // group_size
                down_blocks.push(DownBlock::Down {
                    conv: Conv::load_c(ckpt, &format!("{p}.conv"), device, true, DType::F32)?,
                    sd, sh, sw, out, g,
                });
                ch = out;
            }
        }
        Ok(Self {
            mean: stat("mean-of-means")?,
            std: stat("std-of-means")?,
            conv_in: Conv::load_c(ckpt, &format!("{enc}.conv_in"), device, true, DType::F32)?,
            down_blocks,
            conv_out: Conv::load_c(ckpt, &format!("{enc}.conv_out"), device, true, DType::F32)?,
            patch_size: cfg.patch_size,
            device,
        })
    }

    /// Кодировать кадры `[B,3,F,H,W]` ([−1,1]) → латент `[B,128,F',H',W']`.
    /// F кропится до 1+8k. Возвращает нормализованные means.
    pub fn encode(&self, frames: &Tensor) -> Result<Tensor, LtxError> {
        let f = frames.dims()[2];
        let crop = (f - 1) % 8;
        let frames = if crop != 0 { frames.narrow(2, 0, f - crop)?.contiguous()? } else { frames.clone() };
        let mut x = frames.to_device(self.device)?.to_dtype(DType::F32)?;
        x = patchify(&x, self.patch_size)?;
        x = self.conv_in.fwd(&x)?;
        for blk in &self.down_blocks {
            x = match blk {
                DownBlock::Res(res) => {
                    let mut h = x;
                    for r in res {
                        h = r.forward(&h)?;
                    }
                    h
                }
                DownBlock::Down { conv, sd, sh, sw, out, g } => {
                    // causal-pad по времени при sd=2 (дублируем первый кадр)
                    let xp = if *sd == 2 {
                        let first = x.narrow(2, 0, 1)?.contiguous()?;
                        Tensor::cat(&[&first, &x], 2)?.contiguous()?
                    } else {
                        x.clone()
                    };
                    // skip: space_to_depth → group-mean
                    let s2d = space_to_depth(&xp, *sd, *sh, *sw)?;
                    let (b, _cc, d, h, w) = (s2d.dims()[0], s2d.dims()[1], s2d.dims()[2], s2d.dims()[3], s2d.dims()[4]);
                    let x_in = if *g > 1 {
                        s2d.reshape(vec![b, *out, *g, d, h, w])?.mean_keepdim(2)?.reshape(vec![b, *out, d, h, w])?
                    } else {
                        s2d
                    };
                    // conv → space_to_depth
                    let c = conv.fwd(&xp)?;
                    let cs = space_to_depth(&c, *sd, *sh, *sw)?;
                    cs.add(&x_in)?
                }
            };
        }
        let x = pixel_norm_act(&x, true)?;
        let x = self.conv_out.fwd(&x)?; // [B,129,F',H',W']
        // UNIFORM: means = первые 128 каналов; normalize (means − mean)/std
        let means = x.narrow(1, 0, 128)?.contiguous()?;
        Ok(means.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?)
    }
}
