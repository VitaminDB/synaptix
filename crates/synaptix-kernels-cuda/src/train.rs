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
        let new = Arc::new(Self { module });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    fn f(&self, name: &str) -> Result<CudaFunction> {
        load_fn(&self.module, name)
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
    }
    Ok(())
}
