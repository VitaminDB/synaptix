//! Ядра обучения ([`TrainOp`]): обратные проходы RMSNorm и SwiGLU, слитая
//! cross-entropy, каузальный softmax с упаковкой примеров, деквант
//! перемешанного NVFP4, AdamW. Исходник — `cu/train/train.cu`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use parking_lot::Mutex;
use synaptix_core::backend::train::TrainOp;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::Storage;

use crate::kernels::compile::{compile_module, load_fn};

pub struct TrainKernels {
    module: Arc<CudaModule>,
    flash: Arc<CudaModule>,
    gdn: Arc<CudaModule>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<TrainKernels>)>>> = OnceLock::new();

impl TrainKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        if let Some((_, v)) = cache.lock().iter().find(|(k, _)| *k == key) {
            return Ok(v.clone());
        }
        let module = compile_module(ctx, include_str!("cu/train/train.cu"), "train.cu")?;
        let flash = compile_module(ctx, include_str!("cu/train/flash_train.cu"), "flash_train.cu")?;
        let gdn = compile_module(ctx, include_str!("cu/train/gdn_train.cu"), "gdn_train.cu")?;
        let new = Arc::new(Self { module, flash, gdn });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    fn f(&self, name: &str) -> Result<CudaFunction> {
        load_fn(&self.module, name)
    }

    fn gdn_fn(&self, name: &str) -> Result<CudaFunction> {
        load_fn(&self.gdn, name)
    }

    fn flash_fn(&self, name: &str, smem: usize) -> Result<CudaFunction> {
        let f = load_fn(&self.flash, name)?;
        f.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem as i32,
        )
        .map_err(|e| SynaptixError::Cuda(format!("set smem {name}: {e:?}")))?;
        Ok(f)
    }
}

fn sfx(dt: DType) -> Result<&'static str> {
    Ok(match dt {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        _ => return Err(SynaptixError::Unsupported("train_op: тип входа не F32/BF16/F16")),
    })
}

fn input<'a>(ins: &'a [(&Storage, &Layout)], i: usize, what: &'static str) -> Result<(&'a CudaSlice<u8>, &'a Layout)> {
    let (st, lo) = ins.get(i).ok_or(SynaptixError::Unsupported(what))?;
    let buf = st.as_cuda().ok_or(SynaptixError::Unsupported("train_op: вход не на CUDA"))?;
    if !lo.is_contiguous() || lo.byte_offset() != 0 {
        return Err(SynaptixError::NonContiguous);
    }
    Ok((buf.slice(), lo))
}

fn rows_cfg(rows: usize, threads: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

fn flat_cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u64).div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

fn expect_dt(lo: &Layout, dt: DType, what: &'static str) -> Result<()> {
    if lo.dtype() != dt {
        return Err(SynaptixError::Unsupported(what));
    }
    Ok(())
}

fn launch_err(name: &str) -> impl Fn(cudarc::driver::DriverError) -> SynaptixError + '_ {
    move |e| SynaptixError::Cuda(format!("launch {name}: {e:?}"))
}

pub fn run_train_op(
    op: TrainOp,
    ins: &[(&Storage, &Layout)],
    outs: &mut [(&mut Storage, &Layout)],
) -> Result<()> {
    let (first, _) = ins.first().ok_or(SynaptixError::Unsupported("train_op: нет входов"))?;
    let cb = first.as_cuda().ok_or(SynaptixError::Unsupported("train_op: вход не на CUDA"))?;
    let ctx = cb.device().clone();
    let stream = synaptix_core::device::cuda::default_stream(cb.ordinal())?;
    let k = TrainKernels::for_context(&ctx)?;
    let out_layouts: Vec<Layout> = outs.iter().map(|(_, l)| (*l).clone()).collect();
    let mut out_bufs: Vec<&mut CudaSlice<u8>> = Vec::with_capacity(outs.len());
    for (st, _) in outs.iter_mut() {
        let b = st.as_cuda_mut().ok_or(SynaptixError::Unsupported("train_op: выход не на CUDA"))?;
        out_bufs.push(b.slice_mut());
    }
    let need_outs = |n: usize| -> Result<()> {
        if out_bufs.len() < n {
            return Err(SynaptixError::Unsupported("train_op: мало выходов"));
        }
        Ok(())
    };

    match op {
        TrainOp::RmsNormBwd { eps } => {
            let (x, xl) = input(ins, 0, "RmsNormBwd: x")?;
            let (w, wl) = input(ins, 1, "RmsNormBwd: w")?;
            let (dy, dyl) = input(ins, 2, "RmsNormBwd: dy")?;
            need_outs(1)?;
            if wl.dtype() != xl.dtype() {
                return Err(SynaptixError::Unsupported("RmsNormBwd: x и w разного типа"));
            }
            expect_dt(dyl, DType::F32, "RmsNormBwd: dy не F32")?;
            expect_dt(&out_layouts[0], DType::F32, "RmsNormBwd: dx не F32")?;
            let h = *xl.dims().last().unwrap();
            let rows = xl.numel() / h;
            let name = format!("rms_norm_bwd_{}", sfx(xl.dtype())?);
            let f = k.f(&name)?;
            let (hi, e) = (h as i32, eps);
            let mut b = stream.launch_builder(&f);
            b.arg(x).arg(w).arg(dy).arg(&mut *out_bufs[0]).arg(&hi).arg(&e);
            unsafe { b.launch(rows_cfg(rows, 256)) }.map_err(launch_err(&name))?;
        }
        TrainOp::SiluMulBwd => {
            let (g, gl) = input(ins, 0, "SiluMulBwd: gate")?;
            let (u, ul) = input(ins, 1, "SiluMulBwd: up")?;
            let (dy, dyl) = input(ins, 2, "SiluMulBwd: dy")?;
            need_outs(2)?;
            if gl.dtype() != ul.dtype() {
                return Err(SynaptixError::Unsupported("SiluMulBwd: gate и up разного типа"));
            }
            expect_dt(dyl, DType::F32, "SiluMulBwd: dy не F32")?;
            let n = gl.numel();
            let name = format!("silu_mul_bwd_{}", sfx(gl.dtype())?);
            let f = k.f(&name)?;
            let ni = n as i64;
            let (o0, o1) = out_bufs.split_at_mut(1);
            let mut b = stream.launch_builder(&f);
            b.arg(g).arg(u).arg(dy).arg(&mut *o0[0]).arg(&mut *o1[0]).arg(&ni);
            unsafe { b.launch(flat_cfg(n)) }.map_err(launch_err(&name))?;
        }
        TrainOp::CrossEntropy { scale } => {
            let (x, xl) = input(ins, 0, "CrossEntropy: logits")?;
            let (lab, ll) = input(ins, 1, "CrossEntropy: labels")?;
            need_outs(2)?;
            expect_dt(ll, DType::I32, "CrossEntropy: labels не I32")?;
            let v = *xl.dims().last().unwrap();
            let rows = xl.numel() / v;
            let name = format!("ce_fwd_bwd_{}_{}", sfx(xl.dtype())?, sfx(out_layouts[1].dtype())?);
            let f = k.f(&name)?;
            let (vi, s) = (v as i32, scale);
            let (o0, o1) = out_bufs.split_at_mut(1);
            let mut b = stream.launch_builder(&f);
            b.arg(x).arg(lab).arg(&mut *o0[0]).arg(&mut *o1[0]).arg(&vi).arg(&s);
            unsafe { b.launch(rows_cfg(rows, 1024)) }.map_err(launch_err(&name))?;
        }
        TrainOp::CausalSoftmax { scale, has_seg } => {
            let (s, sl) = input(ins, 0, "CausalSoftmax: scores")?;
            expect_dt(sl, DType::F32, "CausalSoftmax: scores не F32")?;
            need_outs(1)?;
            let t = *sl.dims().last().unwrap();
            if sl.dims().len() < 2 || sl.dims()[sl.dims().len() - 2] != t {
                return Err(SynaptixError::Unsupported("CausalSoftmax: ожидались скоры [B, T, T]"));
            }
            let rows = sl.numel() / t;
            let seg = if has_seg { Some(input(ins, 1, "CausalSoftmax: seg")?.0) } else { None };
            let f = k.f("causal_softmax_f32")?;
            let (ti, sc, hs) = (t as i32, scale, has_seg as i32);
            let mut b = stream.launch_builder(&f);
            b.arg(s);
            match seg {
                Some(sg) => b.arg(sg),
                None => b.arg(s),
            };
            b.arg(&mut *out_bufs[0]).arg(&ti).arg(&sc).arg(&hs);
            unsafe { b.launch(rows_cfg(rows, 256)) }.map_err(launch_err("causal_softmax_f32"))?;
        }
        TrainOp::SoftmaxBwd { scale } => {
            let (p, pl) = input(ins, 0, "SoftmaxBwd: p")?;
            let (dp, dpl) = input(ins, 1, "SoftmaxBwd: dp")?;
            expect_dt(pl, DType::F32, "SoftmaxBwd: p не F32")?;
            expect_dt(dpl, DType::F32, "SoftmaxBwd: dp не F32")?;
            need_outs(1)?;
            let l = *pl.dims().last().unwrap();
            let rows = pl.numel() / l;
            let f = k.f("softmax_bwd_f32")?;
            let (li, sc) = (l as i32, scale);
            let mut b = stream.launch_builder(&f);
            b.arg(p).arg(dp).arg(&mut *out_bufs[0]).arg(&li).arg(&sc);
            unsafe { b.launch(rows_cfg(rows, 256)) }.map_err(launch_err("softmax_bwd_f32"))?;
        }
        TrainOp::Nvfp4DequantShuffled { n, k: kk } => {
            let (sh, _) = input(ins, 0, "Nvfp4DequantShuffled: shuffled")?;
            let (sc, _) = input(ins, 1, "Nvfp4DequantShuffled: scales")?;
            need_outs(1)?;
            if n % 16 != 0 || kk % 64 != 0 {
                return Err(SynaptixError::Unsupported("Nvfp4DequantShuffled: n % 16 и k % 64"));
            }
            expect_dt(&out_layouts[0], DType::BF16, "Nvfp4DequantShuffled: выход не BF16")?;
            let f = k.f("nvfp4_dequant_shuf_bf16")?;
            let (ni, ki) = (n as u32, kk as u32);
            let mut b = stream.launch_builder(&f);
            b.arg(sh).arg(sc).arg(&mut *out_bufs[0]).arg(&ni).arg(&ki);
            unsafe { b.launch(flat_cfg(n * kk / 2)) }.map_err(launch_err("nvfp4_dequant_shuf_bf16"))?;
        }
        TrainOp::AdamW { lr, beta1, beta2, eps, weight_decay, bc1, bc2 } => {
            let (p, pl) = input(ins, 0, "AdamW: p")?;
            let (g, _) = input(ins, 1, "AdamW: g")?;
            let (m, _) = input(ins, 2, "AdamW: m")?;
            let (v, _) = input(ins, 3, "AdamW: v")?;
            need_outs(3)?;
            for (_, l) in ins.iter().take(4) {
                expect_dt(l, DType::F32, "AdamW: всё в F32")?;
                if l.numel() != pl.numel() {
                    return Err(SynaptixError::Unsupported("AdamW: разная длина"));
                }
            }
            let n = pl.numel();
            let f = k.f("adamw_f32")?;
            let ni = n as i64;
            let (o0, rest) = out_bufs.split_at_mut(1);
            let (o1, o2) = rest.split_at_mut(1);
            let mut b = stream.launch_builder(&f);
            b.arg(p).arg(g).arg(m).arg(v).arg(&mut *o0[0]).arg(&mut *o1[0]).arg(&mut *o2[0]);
            b.arg(&ni).arg(&lr).arg(&beta1).arg(&beta2).arg(&eps).arg(&weight_decay).arg(&bc1).arg(&bc2);
            unsafe { b.launch(flat_cfg(n)) }.map_err(launch_err("adamw_f32"))?;
        }
        TrainOp::LoraDown => {
            let (x, xl) = input(ins, 0, "LoraDown: x")?;
            let (w, wl) = input(ins, 1, "LoraDown: w")?;
            need_outs(1)?;
            expect_dt(wl, DType::F32, "LoraDown: w не F32")?;
            let (kk, r) = (*xl.dims().last().unwrap(), wl.dims()[0]);
            if r > 64 || wl.dims()[1] != kk {
                return Err(SynaptixError::Unsupported("LoraDown: r ≤ 64, w [r, K]"));
            }
            let t = xl.numel() / kk;
            let name = format!("lora_down_{}", sfx(xl.dtype())?);
            let f = k.f(&name)?;
            let (ti, ki, ri) = (t as i32, kk as i32, r as i32);
            let mut b = stream.launch_builder(&f);
            b.arg(x).arg(w).arg(&mut *out_bufs[0]).arg(&ti).arg(&ki).arg(&ri);
            unsafe { b.launch(rows_cfg(t.div_ceil(4), 256)) }.map_err(launch_err(&name))?;
        }
        TrainOp::LoraUpAdd { scale } => {
            let (base, bl) = input(ins, 0, "LoraUpAdd: base")?;
            let (z, zl) = input(ins, 1, "LoraUpAdd: z")?;
            let (m, ml) = input(ins, 2, "LoraUpAdd: m")?;
            need_outs(1)?;
            expect_dt(zl, DType::F32, "LoraUpAdd: z не F32")?;
            expect_dt(ml, DType::F32, "LoraUpAdd: m не F32")?;
            if out_layouts[0].dtype() != bl.dtype() {
                return Err(SynaptixError::Unsupported("LoraUpAdd: выход другого типа, чем base"));
            }
            let n = *bl.dims().last().unwrap();
            let r = ml.dims()[1];
            if r > 64 || ml.dims()[0] != n || *zl.dims().last().unwrap() != r {
                return Err(SynaptixError::Unsupported("LoraUpAdd: формы"));
            }
            let t = bl.numel() / n;
            let name = format!("lora_up_add_{}", match bl.dtype() { DType::F32 => "f32", DType::BF16 => "bf16", DType::F16 => "f16", _ => return Err(SynaptixError::Unsupported("LoraUpAdd: base F32/BF16/F16")) });
            let f = k.f(&name)?;
            let (ti, ni, ri, s) = (t as i32, n as i32, r as i32, scale);
            let mut b = stream.launch_builder(&f);
            b.arg(base).arg(z).arg(m).arg(&mut *out_bufs[0]).arg(&ti).arg(&ni).arg(&ri).arg(&s);
            let cfg = LaunchConfig { grid_dim: (n.div_ceil(256) as u32, t.div_ceil(4) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(cfg) }.map_err(launch_err(&name))?;
        }
        TrainOp::LoraOuter => {
            let (a, al) = input(ins, 0, "LoraOuter: a")?;
            let (bb, bl) = input(ins, 1, "LoraOuter: b")?;
            need_outs(1)?;
            expect_dt(al, DType::F32, "LoraOuter: a не F32")?;
            let r = *al.dims().last().unwrap();
            let kk = *bl.dims().last().unwrap();
            let t = al.numel() / r;
            if bl.numel() / kk != t || r > 64 {
                return Err(SynaptixError::Unsupported("LoraOuter: формы"));
            }
            let parts = t.div_ceil(128);
            if out_layouts[0].numel() != parts * r * kk {
                return Err(SynaptixError::Unsupported("LoraOuter: выход [⌈T/128⌉, r, K]"));
            }
            let name = format!("lora_outer_{}", sfx(bl.dtype())?);
            let f = k.f(&name)?;
            let (ti, ki, ri) = (t as i32, kk as i32, r as i32);
            let mut b = stream.launch_builder(&f);
            b.arg(a).arg(bb).arg(&mut *out_bufs[0]).arg(&ti).arg(&ki).arg(&ri);
            let cfg = LaunchConfig { grid_dim: (kk.div_ceil(256) as u32, parts as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(cfg) }.map_err(launch_err(&name))?;
        }
        TrainOp::FlashFwd { scale, has_seg } => {
            let (q, ql) = input(ins, 0, "FlashFwd: q")?;
            let (kb, kl) = input(ins, 1, "FlashFwd: k")?;
            let (v, _) = input(ins, 2, "FlashFwd: v")?;
            let seg = if has_seg { Some(input(ins, 3, "FlashFwd: seg")?.0) } else { None };
            need_outs(2)?;
            for l in [ql, kl] {
                expect_dt(l, DType::BF16, "FlashFwd: q/k/v BF16")?;
            }
            let (h, t, d) = (ql.dims()[0], ql.dims()[1], ql.dims()[2]);
            let hkv = kl.dims()[0];
            if hkv == 0 || h % hkv != 0 || kl.dims()[1] != t || kl.dims()[2] != d {
                return Err(SynaptixError::Unsupported("FlashFwd: формы q/k"));
            }
            let (warps, bc) = match d {
                64 | 128 => (4usize, 64usize),
                256 => (2, 32),
                _ => return Err(SynaptixError::Unsupported("FlashFwd: D — 64, 128 или 256")),
            };
            let smem = 2 * bc * d * 2 + warps * 16 * d * 4 + warps * 16 * bc * 4;
            let name = format!("flash_train_fwd_d{d}");
            let f = k.flash_fn(&name, smem)?;
            let (ti, hi, hki, sc) = (t as i32, h as i32, hkv as i32, scale);
            let null: u64 = 0;
            let (o0, o1) = out_bufs.split_at_mut(1);
            let mut b = stream.launch_builder(&f);
            b.arg(q).arg(kb).arg(v);
            match seg {
                Some(sg) => b.arg(sg),
                None => b.arg(&null),
            };
            b.arg(&mut *o0[0]).arg(&mut *o1[0]).arg(&ti).arg(&hi).arg(&hki).arg(&sc);
            let cfg = LaunchConfig {
                grid_dim: (t.div_ceil(warps * 16) as u32, h as u32, 1),
                block_dim: ((warps * 32) as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            unsafe { b.launch(cfg) }.map_err(launch_err(&name))?;
        }
        TrainOp::FlashBwd { scale, has_seg } => {
            let (q, ql) = input(ins, 0, "FlashBwd: q")?;
            let (kb, kl) = input(ins, 1, "FlashBwd: k")?;
            let (v, _) = input(ins, 2, "FlashBwd: v")?;
            let (dout, _) = input(ins, 3, "FlashBwd: dout")?;
            let (lse, _) = input(ins, 4, "FlashBwd: lse")?;
            let (delta, _) = input(ins, 5, "FlashBwd: delta")?;
            let seg = if has_seg { Some(input(ins, 6, "FlashBwd: seg")?.0) } else { None };
            need_outs(3)?;
            let (h, t, d) = (ql.dims()[0], ql.dims()[1], ql.dims()[2]);
            let hkv = kl.dims()[0];
            let tp = t.div_ceil(32) * 32;
            if out_layouts[1].numel() != hkv * tp * d || out_layouts[2].numel() != hkv * tp * d {
                return Err(SynaptixError::Unsupported("FlashBwd: dk/dv — [Hkv, ⌈T/32⌉·32, D]"));
            }
            if !matches!(d, 64 | 128 | 256) {
                return Err(SynaptixError::Unsupported("FlashBwd: D — 64, 128 или 256"));
            }
            for ob in out_bufs.iter_mut() {
                stream.memset_zeros(&mut **ob).map_err(|e| SynaptixError::Cuda(format!("FlashBwd: memset: {e:?}")))?;
            }
            let smem = 4 * 32 * d * 2 + 2 * 32 * 4 + 2 * 16 * 32 * 4 + 2 * 2 * 16 * 32 * 2 + 2 * 256 * 4;
            let name = format!("flash_train_bwd_d{d}");
            let f = k.flash_fn(&name, smem)?;
            let (ti, tpi, hi, hki, sc) = (t as i32, tp as i32, h as i32, hkv as i32, scale);
            let null: u64 = 0;
            let (o0, rest) = out_bufs.split_at_mut(1);
            let (o1, o2) = rest.split_at_mut(1);
            let mut b = stream.launch_builder(&f);
            b.arg(q).arg(kb).arg(v);
            match seg {
                Some(sg) => b.arg(sg),
                None => b.arg(&null),
            };
            b.arg(dout).arg(lse).arg(delta).arg(&mut *o0[0]).arg(&mut *o1[0]).arg(&mut *o2[0]);
            b.arg(&ti).arg(&tpi).arg(&hi).arg(&hki).arg(&sc);
            let cfg = LaunchConfig { grid_dim: (tp as u32 / 32, hkv as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: smem as u32 };
            unsafe { b.launch(cfg) }.map_err(launch_err(&name))?;
        }
        TrainOp::GdnConvFwd { has_seg } | TrainOp::GdnConvBwd { has_seg } => {
            let fwd = matches!(op, TrainOp::GdnConvFwd { .. });
            let (x, xl) = input(ins, 0, "GdnConv: x")?;
            let (w, wl) = input(ins, 1, "GdnConv: w")?;
            let seg = if has_seg { Some(input(ins, 2, "GdnConv: seg")?.0) } else { None };
            need_outs(1)?;
            expect_dt(wl, DType::F32, "GdnConv: w не F32")?;
            let c = *xl.dims().last().unwrap();
            let t = xl.numel() / c;
            let kk = wl.dims()[wl.dims().len() - 1];
            if wl.numel() != c * kk {
                return Err(SynaptixError::Unsupported("GdnConv: w [C, K]"));
            }
            let name = if fwd { format!("gdn_conv_fwd_{}", sfx(xl.dtype())?) } else { "gdn_conv_bwd".to_string() };
            if !fwd {
                expect_dt(xl, DType::F32, "GdnConvBwd: dy не F32")?;
            }
            let f = k.gdn_fn(&name)?;
            let (ti, ci, ki) = (t as i32, c as i32, kk as i32);
            let null: u64 = 0;
            let mut b = stream.launch_builder(&f);
            b.arg(x).arg(w);
            match seg {
                Some(sg) => b.arg(sg),
                None => b.arg(&null),
            };
            b.arg(&mut *out_bufs[0]).arg(&ti).arg(&ci).arg(&ki);
            unsafe { b.launch(flat_cfg(t * c)) }.map_err(launch_err(&name))?;
        }
        TrainOp::GdnFwd { has_seg } => {
            let (q, ql) = input(ins, 0, "GdnFwd: q")?;
            let (kb, _) = input(ins, 1, "GdnFwd: k")?;
            let (v, vl) = input(ins, 2, "GdnFwd: v")?;
            let (g, _) = input(ins, 3, "GdnFwd: g")?;
            let (be, _) = input(ins, 4, "GdnFwd: beta")?;
            let seg = if has_seg { Some(input(ins, 5, "GdnFwd: seg")?.0) } else { None };
            need_outs(2)?;
            let (t, hk, dk) = (ql.dims()[0], ql.dims()[1], ql.dims()[2]);
            let (hv, dv) = (vl.dims()[1], vl.dims()[2]);
            if dk != 128 || dv % 32 != 0 || hv % hk != 0 {
                return Err(SynaptixError::Unsupported("GdnFwd: DK = 128, dv кратно 32"));
            }
            let f = k.gdn_fn("gdn_fwd")?;
            let (ti, hki, hvi, dvi) = (t as i32, hk as i32, hv as i32, dv as i32);
            let null: u64 = 0;
            let (o0, o1) = out_bufs.split_at_mut(1);
            let mut b = stream.launch_builder(&f);
            b.arg(q).arg(kb).arg(v).arg(g).arg(be);
            match seg {
                Some(sg) => b.arg(sg),
                None => b.arg(&null),
            };
            b.arg(&mut *o0[0]).arg(&mut *o1[0]).arg(&ti).arg(&hki).arg(&hvi).arg(&dvi);
            let cfg = LaunchConfig { grid_dim: ((dv / 32) as u32, hv as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(cfg) }.map_err(launch_err("gdn_fwd"))?;
        }
        TrainOp::GdnBwd { has_seg } => {
            let (q, ql) = input(ins, 0, "GdnBwd: q")?;
            let (kb, _) = input(ins, 1, "GdnBwd: k")?;
            let (v, vl) = input(ins, 2, "GdnBwd: v")?;
            let (g, _) = input(ins, 3, "GdnBwd: g")?;
            let (be, _) = input(ins, 4, "GdnBwd: beta")?;
            let (ck, _) = input(ins, 5, "GdnBwd: ckpt")?;
            let (dout, _) = input(ins, 6, "GdnBwd: dout")?;
            let seg = if has_seg { Some(input(ins, 7, "GdnBwd: seg")?.0) } else { None };
            need_outs(6)?;
            let (t, hk, dk) = (ql.dims()[0], ql.dims()[1], ql.dims()[2]);
            let (hv, dv) = (vl.dims()[1], vl.dims()[2]);
            if dk != 128 || dv % 32 != 0 || hv % hk != 0 {
                return Err(SynaptixError::Unsupported("GdnBwd: DK = 128, dv кратно 32"));
            }
            let f = k.gdn_fn("gdn_bwd")?;
            let (ti, hki, hvi, dvi) = (t as i32, hk as i32, hv as i32, dv as i32);
            let null: u64 = 0;
            let (o0, r1) = out_bufs.split_at_mut(1);
            let (o1, r2) = r1.split_at_mut(1);
            let (o2, r3) = r2.split_at_mut(1);
            let (o3, r4) = r3.split_at_mut(1);
            let (o4, o5) = r4.split_at_mut(1);
            let mut b = stream.launch_builder(&f);
            b.arg(q).arg(kb).arg(v).arg(g).arg(be);
            match seg {
                Some(sg) => b.arg(sg),
                None => b.arg(&null),
            };
            b.arg(ck).arg(dout);
            b.arg(&mut *o0[0]).arg(&mut *o1[0]).arg(&mut *o2[0]).arg(&mut *o3[0]).arg(&mut *o4[0]).arg(&mut *o5[0]);
            b.arg(&ti).arg(&hki).arg(&hvi).arg(&dvi);
            let cfg = LaunchConfig { grid_dim: ((dv / 32) as u32, hv as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(cfg) }.map_err(launch_err("gdn_bwd"))?;
        }
    }
    Ok(())
}
