//! Диффузионный видеодекодер LTX-2.5 (`CausalDiffusionVAE` /
//! `NADiffusionDecoder`, файл `ltx-2.5-video-vae`): вместо свёрточного VAE —
//! четыре детерминированные стадии блоков с вниманием по трёхмерной
//! окрестности (NATTEN) и пятая, диффузионная (один шаг, выход x0) над
//! пиксельными патчами 4×4. Эталон: LTX-2 `diffusion_video_decoder.py`.
//!
//! Раскладка channels-last, токен = (t·H + h)·W + w; тензоры — `[1, N, C]`
//! BF16. Внимание — ядра `ltx_na_prep` / `ltx_na3d` (`hy4.cu`), срезами по
//! времени: объём стадии 5 (миллионы токенов) целиком в q/k/v не помещается.

use std::path::Path;

use synaptix_core::backend::{registry, Ink2Launch};
use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_io::weights::WeightLoader;

use crate::text_encoder::linear;
use crate::LtxError;

type R<T> = Result<T, SynaptixError>;

const DT: DType = DType::BF16;
const EPS: f32 = 1e-6;
/// Глубины стадий и ядра окрестности (t, h, w); каналы стадий — 2048, 1024, 512, 512, 256.
const DEPTH: [usize; 5] = [4, 6, 4, 2, 8];
const KERNEL: [(usize, usize, usize); 5] = [(3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (11, 11, 11)];
/// Апсемплы после стадий 1–4: шаги (t, h, w).
const UP: [(usize, usize, usize); 4] = [(1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)];
/// Бюджет элементов временных тензоров построчных операций.
const ROW_BUDGET: usize = 192 << 20;
/// Бюджет токенов среза внимания (q', k', qkv среза).
const SLAB_TOKENS: usize = 1 << 19;
/// Бюджет токенов объёма стадии 5 на срез по времени и перекрытие срезов, кадров.
const TILE_TOKENS: usize = 8 << 20;
const TILE_OVERLAP: usize = 40;

fn addr(t: &Tensor) -> R<u64> {
    if !t.is_contiguous() {
        return Err(SynaptixError::NonContiguous);
    }
    t.device_address().ok_or(SynaptixError::Unsupported("diff_vae: тензор не на карте"))
}

fn launch(dev: Device, name: &str, a: &Ink2Launch, grid: u32, block: u32) -> R<()> {
    registry::backend_for(dev)?.hy4_launch(dev, name, a, (grid.max(1), 1, 1), block)
}

struct Block {
    norm1: Tensor,
    norm2: Tensor,
    qkv_w: Tensor,
    qkv_b: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    /// Диффузионный блок: проекция контекста и модуляция `(1+m0, m1, 1+m3, m4)`.
    ctx: Option<(Tensor, Tensor)>,
    modul: Option<[Tensor; 4]>,
}

pub struct DiffVaeDecoder {
    conv_in: (Tensor, Tensor),
    conv_in_x: (Tensor, Tensor),
    conv_out: (Tensor, Tensor),
    norm_out: Tensor,
    stages: Vec<Vec<Block>>,
    ups: Vec<(Tensor, Tensor)>,
    mean: Tensor,
    std: Tensor,
    device: Device,
}

/// Объём `[1, N, C]` с сеткой (T, H, W).
struct Vol {
    x: Tensor,
    t: usize,
    h: usize,
    w: usize,
}

impl Vol {
    fn c(&self) -> usize {
        self.x.dims()[2]
    }
}

/// Построчная операция кусками: `f([1, n, Cin]) → [1, n, Cout]`.
fn rows(x: &Tensor, c_out: usize, widest: usize, mut f: impl FnMut(&Tensor) -> R<Tensor>) -> R<Tensor> {
    let n = x.dims()[1];
    let step = (ROW_BUDGET / widest.max(1)).max(1024);
    if n <= step {
        return f(x);
    }
    let mut out = Tensor::empty_uninit(vec![1, n, c_out], x.dtype(), x.device())?;
    let mut off = 0;
    while off < n {
        let m = step.min(n - off);
        let y = f(&x.narrow(1, off, m)?.contiguous()?)?;
        out.copy_rows_from(off, &y.contiguous()?)?;
        off += m;
    }
    Ok(out)
}

fn rms(x: &Tensor, w: &Tensor) -> R<Tensor> {
    x.rms_norm_fused(w, EPS, false)
}

impl Block {
    /// Норма с модуляцией диффузионного блока: `rms(x)·(1+m_a) + m_b`.
    fn norm(&self, x: &Tensor, w: &Tensor, m: Option<(&Tensor, &Tensor)>) -> R<Tensor> {
        let y = rms(x, w)?;
        match m {
            Some((sc, sh)) => y.broadcast_mul(sc)?.broadcast_add(sh),
            None => Ok(y),
        }
    }

    /// `x + attn(norm1(x))` — срезами по времени, результат в новый объём:
    /// старый нужен соседним срезам (ключи окна), полноразмерных
    /// промежуточных тензоров нет.
    fn attn_residual(&self, x: &Tensor, t: usize, h: usize, w: usize, k: (usize, usize, usize)) -> R<Tensor> {
        let dev = x.device();
        let c = x.dims()[2];
        let nh = c / 64;
        let hw = h * w;
        if t < k.0 || h < k.1 || w < k.2 {
            return Err(SynaptixError::Other(format!("diff_vae: объём {t}×{h}×{w} меньше окна {k:?}")));
        }
        let m = self.modul.as_ref().map(|m| (&m[0], &m[1]));
        let mut out = Tensor::empty_uninit(vec![1, t * hw, c], DT, dev)?;
        // Запросы — кадры [a, b), ключи — их окна по ролику [lo, hi).
        let per = (SLAB_TOKENS / hw).max(1);
        let mut a = 0;
        while a < t {
            let b = (a + per).min(t);
            let lo = a.saturating_sub(k.0 / 2).min(t - k.0);
            let hi = ((b - 1).saturating_sub(k.0 / 2).min(t - k.0) + k.0).max(b);
            let tl = hi - lo;
            let nl = tl * hw;
            let xs = x.narrow(1, lo * hw, nl)?.contiguous()?;
            let qkv = rows(&xs, 3 * c, 3 * c + 2 * c, |r| linear(&self.norm(r, &self.norm1, m)?, &self.qkv_w, Some(&self.qkv_b)))?.contiguous()?;
            let q = Tensor::empty_uninit(vec![nl, nh, 64], DT, dev)?;
            let kk = Tensor::empty_uninit(vec![nl, nh, 64], DT, dev)?;
            let mut la = Ink2Launch::default();
            la.a[0] = addr(&qkv)?;
            la.a[1] = addr(&self.q_norm)?;
            la.a[2] = addr(&self.k_norm)?;
            la.a[3] = addr(&q)?;
            la.a[4] = addr(&kk)?;
            la.u[0] = tl as u32;
            la.u[1] = h as u32;
            la.u[2] = w as u32;
            la.u[3] = nh as u32;
            la.f[0] = EPS;
            la.f[1] = 0.125;
            launch(dev, "ltx_na_prep", &la, (nl * nh * 2).div_ceil(256) as u32, 256)?;
            let nq = (b - a) * hw;
            let o = Tensor::empty_uninit(vec![1, nq, c], DT, dev)?;
            let pairs = nq * nh;
            let mut na = Ink2Launch::default();
            na.a[0] = addr(&q)?;
            na.a[1] = addr(&kk)?;
            na.a[2] = addr(&qkv)?;
            na.a[3] = addr(&o)?;
            na.u[0] = tl as u32;
            na.u[1] = h as u32;
            na.u[2] = w as u32;
            na.u[3] = nh as u32;
            na.u[4] = k.0 as u32;
            na.u[5] = k.1 as u32;
            na.u[6] = k.2 as u32;
            na.u[9] = lo as u32;
            na.u[10] = t as u32;
            na.u[11] = (a - lo) as u32;
            na.u[12] = (b - a) as u32;
            // 8 варпов в блоке; запуск кусками.
            let mut base = 0usize;
            while base < pairs {
                let chunk = (pairs - base).min(8 << 20);
                na.u[7] = base as u32;
                na.u[8] = (base as u64 >> 32) as u32;
                launch(dev, "ltx_na3d", &na, chunk.div_ceil(8) as u32, 256)?;
                base += chunk;
            }
            drop((q, kk, qkv));
            let xq = xs.narrow(1, (a - lo) * hw, nq)?.contiguous()?;
            let res = rows(&o, c, c * 2, |r| linear(r, &self.proj_w, Some(&self.proj_b)))?;
            out.copy_rows_from(a * hw, &xq.add(&res)?.contiguous()?)?;
            a = b;
        }
        Ok(out)
    }

    /// `x + swiglu(norm2(x))` построчными кусками.
    fn mlp_residual(&self, x: &Tensor) -> R<Tensor> {
        let c = x.dims()[2];
        let m = self.modul.as_ref().map(|m| (&m[2], &m[3]));
        rows(x, c, 4 * c * 3, |r| {
            let y = self.norm(r, &self.norm2, m)?;
            let g = linear(&y, &self.gate, None)?;
            let u = linear(&y, &self.up, None)?;
            let hdn = g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|g| g.mul(&u)))?;
            r.add(&linear(&hdn, &self.down, None)?)
        })
    }

    /// Блок целиком; у диффузионного сначала `x ← x + context_proj(context)`.
    /// Каждый шаг пишет новый объём, прежний сразу освобождается.
    fn forward(&self, v: &mut Vol, k: (usize, usize, usize), context: Option<&Tensor>) -> R<()> {
        let c = v.c();
        if let (Some((cw, cb)), Some(ctx)) = (&self.ctx, context) {
            let n = v.x.dims()[1];
            let step = (ROW_BUDGET / (c * 2)).max(1024);
            let mut out = Tensor::empty_uninit(vec![1, n, c], DT, v.x.device())?;
            let mut off = 0;
            while off < n {
                let m = step.min(n - off);
                let p = linear(&ctx.narrow(1, off, m)?.contiguous()?, cw, Some(cb))?;
                out.copy_rows_from(off, &v.x.narrow(1, off, m)?.contiguous()?.add(&p)?.contiguous()?)?;
                off += m;
            }
            v.x = out;
        }
        v.x = self.attn_residual(&v.x, v.t, v.h, v.w, k)?;
        v.x = self.mlp_residual(&v.x)?;
        Ok(())
    }
}

/// Pixel shuffle channels-last: `[1, N, C·pt·ph·pw]` → сетка ×(pt, ph, pw);
/// индекс канала `((c·pt + it)·ph + ih)·pw + iw`. При шаге по времени 2
/// первый кадр выхода отбрасывается (причинность).
fn shuffle(x: &Tensor, t: usize, h: usize, w: usize, p: (usize, usize, usize)) -> R<Vol> {
    shuffle_opt(x, t, h, w, p, true)
}

/// То же; `drop_first` — отбрасывать ли первый кадр при шаге по времени 2
/// (у среза не с начала ролика — нет).
fn shuffle_opt(x: &Tensor, t: usize, h: usize, w: usize, p: (usize, usize, usize), drop_first: bool) -> R<Vol> {
    let c = x.dims()[2] / (p.0 * p.1 * p.2);
    let y = x
        .reshape(vec![t, h, w, c, p.0, p.1, p.2])?
        .permute(vec![0, 4, 1, 5, 2, 6, 3])?
        .contiguous()?
        .reshape(vec![1, t * p.0 * h * p.1 * w * p.2, c])?;
    let (mut t2, h2, w2) = (t * p.0, h * p.1, w * p.2);
    let x = if p.0 == 2 && drop_first {
        t2 -= 1;
        y.narrow(1, h2 * w2, t2 * h2 * w2)?.contiguous()?
    } else {
        y
    };
    Ok(Vol { x, t: t2, h: h2, w: w2 })
}

impl DiffVaeDecoder {
    /// Есть ли в файле диффузионный декодер.
    pub fn present(path: &Path) -> bool {
        crate::loader::open_weights(path).is_ok_and(|l| l.contains("decoder.diff_blocks.0.scale_shift_table"))
    }

    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self, LtxError> {
        let ld = crate::loader::open_weights(path.as_ref())?.with_device(device);
        let g = |n: &str, dt: DType| -> Result<Tensor, LtxError> { ld.load_to(n, device, dt).map_err(|e| LtxError::Load(format!("{n}: {e}"))) };
        let b = |n: &str| g(&format!("decoder.{n}"), DT);
        let f = |n: &str| g(&format!("decoder.{n}"), DType::F32);
        // Модуляция стадии 5 при t = 1 (один шаг) — константа: считается один раз.
        let host = |t: &Tensor| -> Result<Vec<f32>, LtxError> { Ok(t.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?) };
        let half = 128usize;
        let mut emb = vec![0f32; 256];
        for i in 0..half {
            let a = 1000.0f32 * (-(10000f32.ln()) * i as f32 / half as f32).exp();
            emb[i] = a.cos();
            emb[half + i] = a.sin();
        }
        let e = Tensor::from_vec(emb, vec![1, 256], device)?;
        let e = linear(&e, &f("t_embedder.mlp.0.weight")?, Some(&f("t_embedder.mlp.0.bias")?))?.silu()?;
        let e = linear(&e, &f("t_embedder.mlp.2.weight")?, Some(&f("t_embedder.mlp.2.bias")?))?.silu()?;
        let modulation = host(&linear(&e, &f("shared_adaln.proj.weight")?, Some(&f("shared_adaln.proj.bias")?))?)?;
        let mut stages = Vec::with_capacity(5);
        for s in 0..5 {
            let mut blocks = Vec::with_capacity(DEPTH[s]);
            for i in 0..DEPTH[s] {
                let p = if s < 4 { format!("det_stages.{s}.{i}") } else { format!("diff_blocks.{i}") };
                let (ctx, modul) = if s == 4 {
                    let sst = host(&f(&format!("{p}.scale_shift_table"))?)?;
                    let row = |j: usize, one: f32| -> Result<Tensor, LtxError> {
                        let v: Vec<f32> = (0..256).map(|c| modulation[j * 256 + c] + sst[j * 256 + c] + one).collect();
                        Ok(Tensor::from_vec(v, vec![1, 1, 256], device)?.to_dtype(DT)?)
                    };
                    (Some((b(&format!("{p}.context_proj.weight"))?, b(&format!("{p}.context_proj.bias"))?)), Some([row(0, 1.0)?, row(1, 0.0)?, row(3, 1.0)?, row(4, 0.0)?]))
                } else {
                    (None, None)
                };
                blocks.push(Block {
                    norm1: b(&format!("{p}.norm1.weight"))?,
                    norm2: b(&format!("{p}.norm2.weight"))?,
                    qkv_w: b(&format!("{p}.attn.qkv.weight"))?,
                    qkv_b: b(&format!("{p}.attn.qkv.bias"))?,
                    q_norm: f(&format!("{p}.attn.q_norm.weight"))?,
                    k_norm: f(&format!("{p}.attn.k_norm.weight"))?,
                    proj_w: b(&format!("{p}.attn.proj.weight"))?,
                    proj_b: b(&format!("{p}.attn.proj.bias"))?,
                    gate: b(&format!("{p}.mlp.w_gate.weight"))?,
                    up: b(&format!("{p}.mlp.w_up.weight"))?,
                    down: b(&format!("{p}.mlp.w_down.weight"))?,
                    ctx,
                    modul,
                });
            }
            stages.push(blocks);
        }
        let mut ups = Vec::with_capacity(4);
        for i in 0..4 {
            ups.push((b(&format!("upsamples.{i}.proj.weight"))?, b(&format!("upsamples.{i}.proj.bias"))?));
        }
        Ok(Self {
            conv_in: (b("conv_in.weight")?, b("conv_in.bias")?),
            conv_in_x: (b("conv_in_x_t.weight")?, b("conv_in_x_t.bias")?),
            conv_out: (b("conv_out.weight")?, b("conv_out.bias")?),
            norm_out: b("norm_out.weight")?,
            stages,
            ups,
            mean: g("per_channel_statistics.mean-of-means", DT)?.reshape(vec![1, 1, 128])?,
            std: g("per_channel_statistics.std-of-means", DT)?.reshape(vec![1, 1, 128])?,
            device,
        })
    }

    /// Латент `[1, 128, F, H, W]` (нормированный) → RGB `[1, 3, 8F−7, 32H, 32W]`
    /// в [−1, 1]. `seed` — шум диффузионной стадии.
    pub fn decode(&self, latent: &Tensor, seed: u64) -> Result<Tensor, LtxError> {
        let d = latent.dims().to_vec();
        let (f, h, w) = (d[2], d[3], d[4]);
        if h < 7 || w < 7 || f < 3 {
            return Err(LtxError::Load(format!("диффузионный декодер: латент {f}×{h}×{w} меньше минимума 3×7×7")));
        }
        let dev = self.device;
        // channels-last, «призрачный» хвост: последний латентный кадр ещё дважды.
        let x = latent.to_device(dev)?.to_dtype(DT)?.permute(vec![0, 2, 3, 4, 1])?.contiguous()?.reshape(vec![1, f * h * w, 128])?;
        let last = x.narrow(1, (f - 1) * h * w, h * w)?.contiguous()?;
        let x = Tensor::cat(&[&x, &last, &last], 1)?.contiguous()?;
        let x = x.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let x = linear(&x, &self.conv_in.0, Some(&self.conv_in.1))?;
        let mut v = Vol { x, t: f + 2, h, w };
        for s in 0..4 {
            synaptix_core::gen_status::phase(format!("диффузионный декодер: стадия {} из 5", s + 1));
            for blk in &self.stages[s] {
                blk.forward(&mut v, KERNEL[s], None)?;
            }
            if s < 3 {
                let (uw, ub) = &self.ups[s];
                let up_c = uw.dims()[0];
                let y = rows(&v.x, up_c, up_c, |r| linear(r, uw, Some(ub)))?;
                v = shuffle(&y, v.t, v.h, v.w, UP[s])?;
            }
        }
        // Выход стадии 4 до апсемпла: кадр j контекста стадии 5 — кадр j+1
        // апсемпла (первый отброшен), то есть половина кадра (j+1)/2 стадии 4.
        // «Призрачный» хвост срезается: 8F + 9 → 8F − 7 кадров.
        let t5 = 8 * f - 7;
        let (h4, w4, hw4) = (v.h, v.w, v.h * v.w);
        let (gh, gw) = (h4 * 2, w4 * 2);
        let hw = gh * gw;
        let v4 = v.x;
        if t5 < KERNEL[4].0 {
            return Err(LtxError::Load(format!("диффузионный декодер: {t5} кадров меньше окна 11")));
        }
        let (uw, ub) = &self.ups[3];
        let context_for = |p: usize, q: usize| -> R<Tensor> {
            let (a, b) = ((p + 1) / 2, q / 2);
            let part = v4.narrow(1, a * hw4, (b - a + 1) * hw4)?.contiguous()?;
            let up_c = uw.dims()[0];
            let y = rows(&part, up_c, up_c, |r| linear(r, uw, Some(ub)))?;
            let vol = shuffle_opt(&y, b - a + 1, h4, w4, UP[3], false)?;
            vol.x.narrow(1, (p + 1 - 2 * a) * hw, (q - p) * hw)?.contiguous()
        };
        // Стадия 5 — срезами по времени (объём растёт с длиной ролика):
        // перекрытие 40 кадров, линейное смешивание (эталон `diffusion_tiling`).
        let budget_frames = (TILE_TOKENS / hw).max(KERNEL[4].0 + 2 * TILE_OVERLAP);
        let tiles: Vec<(usize, usize)> = if t5 <= budget_frames {
            vec![(0, t5)]
        } else {
            let stride = budget_frames - TILE_OVERLAP;
            let mut v = Vec::new();
            let mut p = 0;
            loop {
                let q = (p + budget_frames).min(t5);
                v.push((q.saturating_sub(budget_frames).min(p), q));
                if q == t5 {
                    break;
                }
                p += stride;
            }
            v
        };
        let mut y = Tensor::empty_uninit(vec![1, t5 * hw, 48], DT, dev)?;
        let mut done = 0usize;
        for (ti, &(p, q)) in tiles.iter().enumerate() {
            synaptix_core::gen_status::phase(format!("диффузионный декодер: стадия 5 из 5, срез {} из {}", ti + 1, tiles.len()));
            let context = context_for(p, q)?;
            // Шум сразу в пространстве патчей 4×4 (48 каналов) → 256.
            let n = (q - p) * hw;
            let mut x5 = Tensor::empty_uninit(vec![1, n, 256], DT, dev)?;
            let step = (ROW_BUDGET / 256).max(1024);
            let (mut off, mut k) = (0usize, 0u64);
            while off < n {
                let m = step.min(n - off);
                // `LTX_DIFFVAE_ZERO_NOISE=1` — нулевой шум (сверка с эталоном без общего генератора).
                let noise = if std::env::var_os("LTX_DIFFVAE_ZERO_NOISE").is_some() {
                    Tensor::zeros(vec![1usize, m, 48], DT, dev)?
                } else {
                    Tensor::randn_seeded(vec![1usize, m, 48], seed.wrapping_add(k + ((ti as u64) << 20)), Device::Cpu)?.to_device(dev)?.to_dtype(DT)?
                };
                x5.copy_rows_from(off, &linear(&noise, &self.conv_in_x.0, Some(&self.conv_in_x.1))?.contiguous()?)?;
                off += m;
                k += 1;
            }
            let mut vol = Vol { x: x5, t: q - p, h: gh, w: gw };
            for blk in &self.stages[4] {
                blk.forward(&mut vol, KERNEL[4], Some(&context))?;
            }
            drop(context);
            // norm_out → conv_out (48 = 3·4·4).
            let yt = rows(&vol.x, 48, 256 * 2, |r| linear(&rms(r, &self.norm_out)?, &self.conv_out.0, Some(&self.conv_out.1)))?;
            drop(vol);
            // Перекрытие с уже записанным — покадровое линейное смешивание.
            let ov = done.saturating_sub(p);
            for j in 0..ov {
                let a = (j + 1) as f32 / (ov + 1) as f32;
                let old = y.narrow(1, (p + j) * hw, hw)?.contiguous()?;
                let new = yt.narrow(1, j * hw, hw)?.contiguous()?;
                let mix = old.mul_scalar(1.0 - a)?.add(&new.mul_scalar(a)?)?;
                y.copy_rows_from((p + j) * hw, &mix.contiguous()?)?;
            }
            if q - p > ov {
                y.copy_rows_from((p + ov) * hw, &yt.narrow(1, ov * hw, (q - p - ov) * hw)?.contiguous()?)?;
            }
            done = q;
        }
        drop(v4);
        // unpatchify: канал (c·4 + dw)·4 + dh.
        let rgb = y
            .reshape(vec![t5, gh, gw, 3, 4, 4])?
            .permute(vec![3, 0, 1, 5, 2, 4])?
            .contiguous()?
            .reshape(vec![1, 3, t5, gh * 4, gw * 4])?;
        Ok(rgb)
    }
}
