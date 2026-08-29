use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::backend::{Backend, BinaryOp, ReduceOp, UnaryOp, registry};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::grad::{self, GradOp, UnaryGradKind};
use crate::stream::Stream;
use crate::tensor::Tensor;
use crate::tensor::broadcast::broadcast_layouts;
use crate::tensor::layout::Layout;
use crate::tensor::quant::QuantWeight;
use crate::tensor::shape::Shape;
use crate::tensor::storage::Storage;

/// Можно ли передать K/V layout в `Backend::flash_attention` без материализации
/// contiguous: rank-4 `[B,nkv,T,hd]` с row-major инкрементом по (T,hd)
/// (stride[3]==1, stride[2]==hd) и регулярным per-head/per-batch шагом, где
/// per-head stride кратен hd. Покрывает contiguous (t_stride==T) И strided view
/// preallocated буфера (t_stride==max_seq). Backend выводит t_stride = stride[1]/hd.
fn kv_layout_passthrough_offset_ok(lo: &Layout) -> bool {
    let d = lo.dims();
    if d.len() != 4 {
        return false;
    }
    let s = lo.strides().as_slice();
    let nkv = d[1] as isize;
    let hd = d[3] as isize;
    hd > 0 && s[3] == 1 && s[2] == hd && s[1] > 0 && s[1] % hd == 0 && s[0] == nkv * s[1]
}

fn kv_layout_passthrough_ok(lo: &Layout) -> bool {
    if lo.offset() != 0 {
        return false;
    }
    let d = lo.dims();
    if d.len() != 4 {
        return false;
    }
    let s = lo.strides().as_slice();
    let nkv = d[1] as isize;
    let hd = d[3] as isize;
    hd > 0 && s[3] == 1 && s[2] == hd && s[1] > 0 && s[1] % hd == 0 && s[0] == nkv * s[1]
}

fn attach_unary_grad(op: UnaryOp, input: &Tensor, output: &mut Tensor) -> Result<()> {
    use UnaryGradKind::*;
    let grad_op = match op {
        // Копия — тождество и по значению, и по градиенту.
        UnaryOp::Identity => GradOp::Identity { input },
        UnaryOp::Neg => GradOp::Neg { input },
        UnaryOp::Abs => GradOp::Unary { input, kind: Abs, alpha: None },
        UnaryOp::Sqrt => GradOp::Unary { input, kind: Sqrt, alpha: None },
        UnaryOp::Sqr => GradOp::Unary { input, kind: Square, alpha: None },
        UnaryOp::Recip => GradOp::Unary { input, kind: Recip, alpha: None },
        UnaryOp::Exp => GradOp::Unary { input, kind: Exp, alpha: None },
        UnaryOp::Log => GradOp::Unary { input, kind: Log, alpha: None },
        UnaryOp::Silu => GradOp::Unary { input, kind: SiLU, alpha: None },
        UnaryOp::GeluTanh => GradOp::Unary { input, kind: GeLUTanh, alpha: None },
        UnaryOp::GeluExact => GradOp::Unary { input, kind: GeLUExact, alpha: None },
        UnaryOp::Tanh => GradOp::Unary { input, kind: Tanh, alpha: None },
        UnaryOp::Affine(mul, add) => GradOp::Affine { input, mul, add },
        UnaryOp::Erf => GradOp::Unary { input, kind: Erf, alpha: None },
        UnaryOp::Sigmoid => GradOp::Unary { input, kind: Sigmoid, alpha: None },
        UnaryOp::Relu => GradOp::Unary { input, kind: Relu, alpha: None },
        UnaryOp::Relu2 => GradOp::Unary { input, kind: Relu2, alpha: None },
        UnaryOp::LeakyRelu(alpha) => GradOp::Unary { input, kind: LeakyRelu, alpha: Some(alpha) },
        UnaryOp::Sign => GradOp::Unary { input, kind: Sign, alpha: None },
        UnaryOp::StepGtZero => GradOp::Unary { input, kind: StepGtZero, alpha: None },
        UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Clamp(_, _) | UnaryOp::Powf(_) => return Ok(()),
        UnaryOp::Round | UnaryOp::Floor | UnaryOp::Ceil => return Ok(()),
    };
    grad::try_attach_grad_fn(grad_op, output)
}

fn attach_binary_grad(op: BinaryOp, lhs: &Tensor, rhs: &Tensor, output: &mut Tensor) -> Result<()> {
    let grad_op = match op {
        BinaryOp::Add => GradOp::Add { lhs, rhs },
        BinaryOp::Sub => GradOp::Sub { lhs, rhs },
        BinaryOp::Mul => GradOp::Mul { lhs, rhs },
        BinaryOp::Div => GradOp::Div { lhs, rhs },
        BinaryOp::Max | BinaryOp::Min => return Ok(()),
    };
    grad::try_attach_grad_fn(grad_op, output)
}

fn attach_reduce_grad(
    op: ReduceOp,
    input: &Tensor,
    dims: Vec<usize>,
    keepdim: bool,
    output: &mut Tensor,
) -> Result<()> {
    let grad_op = match op {
        ReduceOp::Sum => GradOp::Sum { input, dims, keepdim },
        ReduceOp::Mean => GradOp::Mean { input, dims, keepdim },
        ReduceOp::Max => GradOp::Max { input, dims, keepdim },
        ReduceOp::ArgMax => return Ok(()),
    };
    grad::try_attach_grad_fn(grad_op, output)
}

#[allow(dead_code)]
pub(crate) fn run_unary(t: &Tensor, op: UnaryOp) -> Result<Tensor> {
    let backend = registry::backend_for(t.device())?;
    let src = t.contiguous_view()?;
    let out_layout = Layout::contiguous(src.shape().clone(), src.dtype());
    let out_bytes = src.dtype().bytes_for_numel(out_layout.numel());
    let mut storage = backend.alloc_uninit(out_bytes, t.device())?;
    let stream = Stream::default_for(t.device())?;
    backend.unary(op, (&src.storage, &src.layout), (&mut storage, &out_layout), &stream)?;
    let mut output = Tensor::from_parts(Arc::new(storage), out_layout);
    attach_unary_grad(op, t, &mut output)?;
    Ok(output)
}

#[allow(dead_code)]
pub(crate) fn run_binary(a: &Tensor, b: &Tensor, op: BinaryOp) -> Result<Tensor> {
    if a.device() != b.device() {
        return Err(SynaptixError::device_mismatch(a.device(), b.device()));
    }
    if a.dtype() != b.dtype() {
        return Err(SynaptixError::dtype_mismatch(a.dtype(), b.dtype()));
    }
    let backend = registry::backend_for(a.device())?;
    let (la, lb, out_shape) = broadcast_layouts(&a.layout, &b.layout)?;
    let out_layout = Layout::contiguous(out_shape, a.dtype());
    let out_bytes = a.dtype().bytes_for_numel(out_layout.numel());
    let mut storage = backend.alloc_uninit(out_bytes, a.device())?;
    let stream = Stream::default_for(a.device())?;
    backend.binary(
        op,
        (&a.storage, &la),
        (&b.storage, &lb),
        (&mut storage, &out_layout),
        &stream,
    )?;
    let mut output = Tensor::from_parts(Arc::new(storage), out_layout);
    attach_binary_grad(op, a, b, &mut output)?;
    Ok(output)
}

#[allow(dead_code)]
pub(crate) fn run_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.device() != b.device() {
        return Err(SynaptixError::device_mismatch(a.device(), b.device()));
    }
    if a.dtype() != b.dtype() {
        return Err(SynaptixError::dtype_mismatch(a.dtype(), b.dtype()));
    }
    let a_dims = a.dims();
    let b_dims = b.dims();
    if a_dims.len() < 2 || b_dims.len() < 2 {
        return Err(SynaptixError::RankMismatch {
            expected: 2,
            got: a_dims.len().min(b_dims.len()),
        });
    }
    let m = a_dims[a_dims.len() - 2];
    let k_a = a_dims[a_dims.len() - 1];
    let k_b = b_dims[b_dims.len() - 2];
    let n = b_dims[b_dims.len() - 1];
    if k_a != k_b {
        return Err(SynaptixError::ShapeMismatch {
            expected: vec![m, k_a],
            got: vec![k_b, n],
        });
    }
    let mut out_dims = Vec::with_capacity(a_dims.len().max(b_dims.len()));
    let batch_a = &a_dims[..a_dims.len() - 2];
    let batch_b = &b_dims[..b_dims.len() - 2];
    let batch_shape = crate::tensor::broadcast::broadcast_shape(batch_a, batch_b)?;
    out_dims.extend_from_slice(batch_shape.dims());
    out_dims.push(m);
    out_dims.push(n);

    let a_contig = a.contiguous_view()?;
    let b_contig = b.contiguous_view()?;
    let backend = registry::backend_for(a.device())?;
    let out_layout = Layout::contiguous(Shape::new(out_dims), a.dtype());
    let out_bytes = a.dtype().bytes_for_numel(out_layout.numel());
    let mut storage = backend.alloc_uninit(out_bytes, a.device())?;
    let stream = Stream::default_for(a.device())?;
    backend.matmul(
        (&a_contig.storage, &a_contig.layout),
        (&b_contig.storage, &b_contig.layout),
        (&mut storage, &out_layout),
        &stream,
    )?;
    let mut output = Tensor::from_parts(Arc::new(storage), out_layout);
    grad::try_attach_grad_fn(GradOp::MatMul { lhs: a, rhs: b }, &mut output)?;
    Ok(output)
}

/// `out = x @ wᵀ`, где `w` хранится в натуральном [out, in] layout (как веса
/// `nn::Linear`). На CUDA при M=1 (decode) роутится в backend GEMV-ядро напрямую
/// (без транспонирования веса и без dense-GEMM). Иначе/при grad — общий путь
/// `matmul(wᵀ)`.
type KrscEntry = (std::sync::Weak<crate::tensor::storage::Storage>, Vec<usize>, Tensor);
static KRSC_CACHE: std::sync::OnceLock<
    parking_lot::Mutex<std::collections::HashMap<usize, KrscEntry>>,
> = std::sync::OnceLock::new();
type WkdEntry = (std::sync::Weak<crate::tensor::storage::Storage>, Vec<usize>, Vec<Tensor>);
static WKD_CACHE: std::sync::OnceLock<
    parking_lot::Mutex<std::collections::HashMap<usize, WkdEntry>>,
> = std::sync::OnceLock::new();

/// Принудительная чистка мёртвых записей кэшей conv-фильтров (krsc + kd-слайсы).
/// Записи держат krsc-копии СИЛЬНЫМИ ссылками, а retain срабатывает только при
/// insert — после дропа conv-модели (upscaler/VAE) её фильтры иначе висят в
/// VRAM, пока другая conv-модель не вставит запись (20s stage2 OOM).
pub fn conv_filter_cache_gc() {
    if let Some(c) = KRSC_CACHE.get() {
        c.lock().retain(|_, (wk, _, _)| wk.upgrade().is_some());
    }
    if let Some(c) = WKD_CACHE.get() {
        c.lock().retain(|_, (wk, _, _)| wk.upgrade().is_some());
    }
}

/// Полная очистка кэшей conv-фильтров (включая живые веса). Звать перед
/// VRAM-критичной фазой, когда conv-модель уже отработала, но ещё жива
/// (upscaler перед stage2-refine): krsc-дубликаты её весов освобождаются,
/// при следующем conv-вызове кэш наполнится заново.
pub fn conv_filter_cache_clear() {
    if let Some(c) = KRSC_CACHE.get() {
        c.lock().clear();
    }
    if let Some(c) = WKD_CACHE.get() {
        c.lock().clear();
    }
}

#[allow(dead_code)]
fn cached_filter_krsc(weight: &Tensor) -> Result<Tensor> {
    let key = Arc::as_ptr(&weight.storage) as usize;
    let cache = KRSC_CACHE.get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
    {
        let g = cache.lock();
        if let Some((wk, dims, t)) = g.get(&key) {
            if wk.upgrade().is_some_and(|s| Arc::as_ptr(&s) as usize == key)
                && dims.as_slice() == weight.dims()
            {
                return Ok(t.clone());
            }
        }
    }
    let krsc = weight.nchw_to_nhwc()?;
    let mut g = cache.lock();
    // Чистка мёртвых записей: запись живёт ровно пока жив ИСХОДНЫЙ вес.
    // Иначе временные веса (conv3d_via_conv2d делает narrow+contiguous kd-слайс
    // на КАЖДЫЙ вызов → новый указатель → новая запись) копят krsc-копии
    // навсегда: VAE-декод тёк ~3.4GB на каждое окно (OOM 20s-видео).
    g.retain(|_, (wk, _, _)| wk.upgrade().is_some());
    g.insert(
        key,
        (Arc::downgrade(&weight.storage), weight.dims().to_vec(), krsc.clone()),
    );
    Ok(krsc)
}

/// KRSC-слайсы conv3d-веса `[Cout,Cin,Kd,Kh,Kw]` по kd: `[Cout,Kh,Kw,Cin]`×Kd.
/// Кэш по указателю ИСХОДНОГО 5D-веса (стабилен в модели) — narrow-слайсы
/// сами по себе временные и через cached_filter_krsc промахивались каждый
/// вызов (переконвертация на каждый conv2d kd-цикла).
fn cached_conv3d_wkd_krsc(wt: &Tensor, kd_size: usize) -> Result<Vec<Tensor>> {
    let key = Arc::as_ptr(&wt.storage) as usize;
    let cache = WKD_CACHE.get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
    {
        let g = cache.lock();
        if let Some((wk, dims, v)) = g.get(&key) {
            if wk.upgrade().is_some_and(|s| Arc::as_ptr(&s) as usize == key)
                && dims.as_slice() == wt.dims()
                && v.len() == kd_size
            {
                return Ok(v.clone());
            }
        }
    }
    let mut v = Vec::with_capacity(kd_size);
    for kdi in 0..kd_size {
        v.push(wt.narrow(2, kdi, 1)?.squeeze(2)?.contiguous()?.nchw_to_nhwc()?);
    }
    let mut g = cache.lock();
    g.retain(|_, (wk, _, _)| wk.upgrade().is_some());
    g.insert(key, (Arc::downgrade(&wt.storage), wt.dims().to_vec(), v.clone()));
    Ok(v)
}

static FORCE_UNFUSED_LINEAR: AtomicBool = AtomicBool::new(false);

/// Форсит общий matmul-путь для Linear вместо fused `backend.linear`/`linear_epilogue`
/// (best_cu NVRTC-GEMM). Fused-путь расходится с matmul на real-данных, что в
/// хаотично-чувствительном MMDiT-стэке (FLUX, 57 блоков) копится в видимую сетку.
/// matmul-путь bit-совпадает с torch F.linear. Включается FLUX-пайплайном.
pub fn set_force_unfused_linear(v: bool) {
    FORCE_UNFUSED_LINEAR.store(v, Ordering::Relaxed);
}
pub fn force_unfused_linear() -> bool {
    FORCE_UNFUSED_LINEAR.load(Ordering::Relaxed)
}

/// RAII-скоуп для [`set_force_unfused_linear`]: ставит флаг на время жизни, при
/// drop восстанавливает прежнее значение (не протекает на другие модели в одном
/// процессе, корректно при early-return через `?`).
pub struct ForceUnfusedLinearGuard(bool);
impl ForceUnfusedLinearGuard {
    pub fn new(v: bool) -> Self {
        Self(FORCE_UNFUSED_LINEAR.swap(v, Ordering::Relaxed))
    }
}
impl Drop for ForceUnfusedLinearGuard {
    fn drop(&mut self) {
        FORCE_UNFUSED_LINEAR.store(self.0, Ordering::Relaxed);
    }
}

pub(crate) fn run_linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    if x.device() != w.device() {
        return Err(SynaptixError::device_mismatch(x.device(), w.device()));
    }
    if x.dtype() != w.dtype() {
        return Err(SynaptixError::dtype_mismatch(x.dtype(), w.dtype()));
    }
    let xr = x.rank();
    if xr == 0 {
        return Err(SynaptixError::Unsupported("linear: scalar x"));
    }
    let k = x.dims()[xr - 1];
    if w.rank() != 2 || w.dims()[1] != k {
        return Err(SynaptixError::ShapeMismatch {
            expected: vec![w.dims().first().copied().unwrap_or(0), k],
            got: w.dims().to_vec(),
        });
    }
    let n = w.dims()[0];
    let _m = x.numel() / k.max(1);

    // Быстрый backend-путь (CUDA: GEMV для M=1, CUTLASS Linear для M>1) — только
    // в no-grad (inference). На CPU/неподдержке backend вернёт Unsupported → fallback.
    // set_force_unfused_linear форсит общий matmul-путь.
    if !grad::needs_graph(&[x, w]) && !force_unfused_linear() {
        let x_c = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
        let w_c = if w.is_contiguous() { w.clone() } else { w.contiguous()? };
        let mut out_dims = x.dims()[..xr - 1].to_vec();
        out_dims.push(n);
        let out_layout = Layout::contiguous(Shape::new(out_dims), x.dtype());
        let out_bytes = x.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(x.device())?;
        // uninit: backend.linear пишет каждую ячейку выхода (либо возвращает
        // Unsupported ДО записи → буфер дропается); memset на каждый вызов
        // стоил ~3-5мкс хост-латентности — заметно на малых M.
        let mut storage = backend.alloc_uninit(out_bytes, x.device())?;
        let stream = Stream::default_for(x.device())?;
        match backend.linear(
            (&x_c.storage, &x_c.layout),
            (&w_c.storage, &w_c.layout),
            (&mut storage, &out_layout),
            &stream,
        ) {
            Ok(()) => return Ok(Tensor::from_parts(Arc::new(storage), out_layout)),
            // backend не умеет этот случай → общий путь ниже
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
    }

    // Общий путь: x @ wᵀ (grad-tracking сохраняется через matmul/transpose).
    let w_t = w.transpose(0, 1)?.contiguous()?;
    run_matmul(x, &w_t)
}

pub(crate) fn run_linear_epilogue(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    residual: Option<&Tensor>,
) -> Result<Tensor> {
    let epilogue_inputs: Vec<&Tensor> = [Some(x), Some(w), bias, residual]
        .into_iter()
        .flatten()
        .collect();
    if !grad::needs_graph(&epilogue_inputs) && !force_unfused_linear() {
        let xr = x.rank();
        if xr >= 1 && w.rank() == 2 {
            let k = x.dims()[xr - 1];
            let n = w.dims()[0];
            let x_c = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
            let w_c = if w.is_contiguous() { w.clone() } else { w.contiguous()? };
            let mut out_dims = x.dims()[..xr - 1].to_vec();
            out_dims.push(n);
            let out_layout = Layout::contiguous(Shape::new(out_dims), x.dtype());
            let out_bytes = x.dtype().bytes_for_numel(out_layout.numel());
            let backend = registry::backend_for(x.device())?;
            let mut storage = backend.alloc_uninit(out_bytes, x.device())?;
            let stream = Stream::default_for(x.device())?;
            let bias_arg = bias.map(|b| (&*b.storage, &b.layout));
            let res_c = match residual {
                Some(r) if r.is_contiguous() => Some(r.clone()),
                Some(r) => Some(r.contiguous()?),
                None => None,
            };
            let res_arg = res_c.as_ref().map(|r| (&*r.storage, &r.layout));
            let _ = (k,);
            match backend.linear_epilogue(
                (&x_c.storage, &x_c.layout),
                (&w_c.storage, &w_c.layout),
                bias_arg,
                res_arg,
                (&mut storage, &out_layout),
                &stream,
            ) {
                Ok(()) => return Ok(Tensor::from_parts(Arc::new(storage), out_layout)),
                Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
                Err(e) => return Err(e),
            }
        }
    }
    let y = run_linear(x, w)?;
    let y = match bias {
        Some(b) => y.broadcast_add(b)?,
        None => y,
    };
    match residual {
        Some(r) => y.add(r),
        None => Ok(y),
    }
}

/// Квантованный Linear (hardware tensor-core путь): `out[M,N] = x[M,K] @ w[N,K]ᵀ`,
/// где `w` — `QuantWeight` (NVFP4/MXFP8) с отдельными scale-тензорами. Вес здесь предквантован при
/// загрузке, а активация `x` (F16) квантуется на лету в backend.
#[allow(dead_code)]
pub(crate) fn run_linear_quant(x: &Tensor, w: &QuantWeight) -> Result<Tensor> {
    if x.device() != w.device() {
        return Err(SynaptixError::device_mismatch(x.device(), w.device()));
    }
    if !matches!(x.dtype(), DType::F16 | DType::BF16) {
        return Err(SynaptixError::Unsupported(
            "linear_quant: активация x должна быть F16 или BF16",
        ));
    }
    let x_dims = x.dims();
    if x_dims.len() < 2 {
        return Err(SynaptixError::RankMismatch {
            expected: 2,
            got: x_dims.len(),
        });
    }
    let k = x_dims[x_dims.len() - 1];
    if k != w.k() {
        return Err(SynaptixError::ShapeMismatch {
            expected: vec![x_dims[x_dims.len() - 2], k],
            got: vec![w.n(), w.k()],
        });
    }
    let mut out_dims = x_dims.to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = w.n();

    let x_contig = x.contiguous_view()?;
    let backend = registry::backend_for(x.device())?;
    let out_layout = Layout::contiguous(Shape::new(out_dims), x.dtype());
    let out_bytes = x.dtype().bytes_for_numel(out_layout.numel());
    // uninit (как run_linear после 3ec0c052): alloc_zeros делал CE-memset
    // выхода (217MB на 26520) на КАЖДЫЙ вызов; все quant-пути пишут out целиком.
    let mut storage = backend.alloc_uninit(out_bytes, x.device())?;
    let stream = Stream::default_for(x.device())?;
    backend.linear_quant(
        (&x_contig.storage, &x_contig.layout),
        w,
        (&mut storage, &out_layout),
        &stream,
    )?;
    Ok(Tensor::from_parts(Arc::new(storage), out_layout))
}

/// Квантует плотный вес `w[N,K]` (F16) в [`QuantWeight`] NVFP4 на загрузке
/// (one-time). Backend выделяет packed/scales и заполняет их; здесь — обёртка
/// в `QuantWeight`. K должно быть кратно 64, N — 16 (требование GEMM-ядер).
pub(crate) fn run_quantize_nvfp4(w: &Tensor) -> Result<QuantWeight> {
    if w.dtype() != DType::F16 {
        return Err(SynaptixError::Unsupported(
            "quantize_to_nvfp4: вес должен быть F16",
        ));
    }
    if w.rank() != 2 {
        return Err(SynaptixError::RankMismatch { expected: 2, got: w.rank() });
    }
    let n = w.dims()[0];
    let k = w.dims()[1];
    let w_contig = w.contiguous_view()?;
    let backend = registry::backend_for(w.device())?;
    let stream = Stream::default_for(w.device())?;
    let (packed, scales) =
        backend.quantize_nvfp4((&w_contig.storage, &w_contig.layout), n, k, &stream)?;
    QuantWeight::new(Arc::new(packed), Arc::new(scales), DType::NVFP4, n, k)
}

pub(crate) fn run_quantize_mxfp8(w: &Tensor) -> Result<QuantWeight> {
    if w.dtype() != DType::F16 {
        return Err(SynaptixError::Unsupported(
            "quantize_to_mxfp8: вес должен быть F16",
        ));
    }
    if w.rank() != 2 {
        return Err(SynaptixError::RankMismatch { expected: 2, got: w.rank() });
    }
    let n = w.dims()[0];
    let k = w.dims()[1];
    if k % 32 != 0 {
        return Err(SynaptixError::Unsupported(
            "quantize_to_mxfp8: K должно быть кратно 32",
        ));
    }
    let w_contig = w.contiguous_view()?;
    let backend = registry::backend_for(w.device())?;
    let stream = Stream::default_for(w.device())?;
    let (packed, scales) =
        backend.quantize_mxfp8((&w_contig.storage, &w_contig.layout), n, k, &stream)?;
    QuantWeight::new(Arc::new(packed), Arc::new(scales), DType::MXFP8, n, k)
}

#[allow(dead_code)]
const REDUCE_STAGE_MIN: usize = 1 << 16;

fn staged_full_reduce(t: &Tensor, op: ReduceOp) -> Result<Option<Tensor>> {
    let dims = t.dims().to_vec();
    let base = if dims.len() >= 2 {
        let short = dims
            .iter()
            .enumerate()
            .min_by_key(|(_, d)| **d)
            .map(|(i, _)| i)
            .unwrap_or(0);
        run_reduce(t, op, &[short], false)?
    } else {
        let n = dims[0];
        let Some(g) = (1..=12)
            .map(|p| 1usize << p)
            .rev()
            .find(|g| n % g == 0 && n / g > 1)
        else {
            return Ok(None);
        };
        run_reduce(&t.contiguous()?.reshape(vec![g, n / g])?, op, &[0], false)?
    };
    let rest: Vec<usize> = (0..base.rank()).collect();
    Ok(Some(run_reduce(&base, op, &rest, false)?))
}

pub(crate) fn run_reduce(t: &Tensor, op: ReduceOp, dims: &[usize], keepdim: bool) -> Result<Tensor> {
    let rank = t.rank();
    if !keepdim
        && rank > 0
        && dims.len() == rank
        && t.numel() > REDUCE_STAGE_MIN
        && t.device().is_cuda()
        && !matches!(op, ReduceOp::ArgMax)
    {
        if let Some(y) = staged_full_reduce(t, op)? {
            return Ok(y);
        }
    }
    let mut sorted = dims.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    for &d in &sorted {
        if d >= rank {
            return Err(SynaptixError::DimOutOfRange { dim: d, rank });
        }
    }
    let mut out_dims: Vec<usize> = Vec::with_capacity(rank);
    for (i, &d) in t.dims().iter().enumerate() {
        if sorted.contains(&i) {
            if keepdim {
                out_dims.push(1);
            }
        } else {
            out_dims.push(d);
        }
    }
    let out_dtype = match op {
        ReduceOp::ArgMax => DType::U32,
        _ => t.dtype(),
    };
    let out_layout = Layout::contiguous(Shape::new(out_dims), out_dtype);
    let backend = registry::backend_for(t.device())?;
    let src = t.contiguous_view()?;
    let out_bytes = out_dtype.bytes_for_numel(out_layout.numel());
    let mut storage = backend.alloc_zeros(out_bytes, t.device())?;
    let stream = Stream::default_for(t.device())?;
    backend.reduce(op, (&src.storage, &src.layout), (&mut storage, &out_layout), &sorted, &stream)?;
    let mut output = Tensor::from_parts(Arc::new(storage), out_layout);
    attach_reduce_grad(op, t, sorted, keepdim, &mut output)?;
    Ok(output)
}

#[allow(dead_code)]
pub(crate) fn run_cast(t: &Tensor, target: DType) -> Result<Tensor> {
    if t.dtype() == target {
        return Ok(t.clone());
    }
    let backend = registry::backend_for(t.device())?;
    let src = t.contiguous_view()?;
    let out_layout = Layout::contiguous(src.shape().clone(), target);
    let out_bytes = target.bytes_for_numel(out_layout.numel());
    let mut storage = backend.alloc_zeros(out_bytes, t.device())?;
    let stream = Stream::default_for(t.device())?;
    backend.cast((&src.storage, &src.layout), (&mut storage, &out_layout), &stream)?;
    let mut output = Tensor::from_parts(Arc::new(storage), out_layout);
    grad::try_attach_grad_fn(GradOp::Cast { input: t, target_dtype: target }, &mut output)?;
    Ok(output)
}

impl Tensor {
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let backend: &'static dyn Backend = registry::backend_for(self.device())?;
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.copy((&self.storage, &self.layout), (&mut storage, &out_layout), &stream)?;
        let mut out = Tensor::from_parts(Arc::new(storage), out_layout);
        grad::try_attach_grad_fn(GradOp::Identity { input: self }, &mut out)?;
        Ok(out)
    }

    fn contiguous_view(&self) -> Result<Self> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            self.contiguous()
        }
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> { run_cast(self, dtype) }

    /// `out = self @ weightᵀ`, `weight` в [out, in] layout (как `nn::Linear`).
    /// CUDA decode (M=1) → GEMV-ядро напрямую; иначе общий `matmul(wᵀ)`.
    pub fn linear(&self, weight: &Tensor) -> Result<Self> { run_linear(self, weight) }

    pub fn linear_bias_residual(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        residual: Option<&Tensor>,
    ) -> Result<Self> {
        run_linear_epilogue(self, weight, bias, residual)
    }

    /// Fused pointwise `out = silu(self) * up` через backend. Возвращает
    /// `Unsupported`, если backend не умеет — вызывающий код должен fallback
    /// на decomposed `self.silu()? .mul(up)`.
    pub fn silu_and_mul(&self, up: &Tensor) -> Result<Self> {
        if self.dtype() != up.dtype() {
            return Err(SynaptixError::Unsupported("silu_and_mul: dtype mismatch"));
        }
        if self.shape() != up.shape() {
            return Err(SynaptixError::Unsupported("silu_and_mul: shape mismatch"));
        }
        let gate = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let upc = if up.is_contiguous() { up.clone() } else { up.contiguous()? };
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.silu_and_mul(
            (&gate.storage, &gate.layout),
            (&upc.storage, &upc.layout),
            (&mut storage, &out_layout),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fused RMSNorm по last dim через backend (один kernel-launch). Возвращает
    /// `Unsupported`, если backend не умеет (CPU/прочие) — вызывающий код
    /// (`synaptix-ops::rms_norm`) тогда падает в decomposed путь.
    pub fn rms_norm_fused(&self, weight: &Tensor, eps: f32, qwen_gain: bool) -> Result<Self> {
        let xr = self.rank();
        if xr == 0 {
            return Err(SynaptixError::Unsupported("rms_norm_fused: scalar x"));
        }
        let h = self.dims()[xr - 1];
        if weight.rank() != 1 || weight.dims()[0] != h {
            return Err(SynaptixError::Unsupported("rms_norm_fused: weight shape"));
        }
        if self.dtype() != weight.dtype() {
            return Err(SynaptixError::Unsupported("rms_norm_fused: dtype mismatch"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let w = if weight.is_contiguous() { weight.clone() } else { weight.contiguous()? };
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.rms_norm(
            (&x.storage, &x.layout),
            (&w.storage, &w.layout),
            (&mut storage, &out_layout),
            eps,
            qwen_gain,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fused residual+RMSNorm: возвращает `(hidden, normed)`, где
    /// `hidden = self + residual` и `normed = RMSNorm(hidden) * weight`. Один
    /// kernel-launch вместо `add` + `rms_norm` (pre-norm transformer-блок:
    /// `hidden += sublayer_out; h = norm(hidden)`). `self`/`residual` `[.., H]`
    /// одной формы; `weight` `[H]`. Падает `Unsupported`, если backend не поддержал.
    pub fn rms_norm_residual_fused(
        &self,
        residual: &Tensor,
        weight: &Tensor,
        eps: f32,
        qwen_gain: bool,
    ) -> Result<(Self, Self)> {
        let xr = self.rank();
        if xr == 0 {
            return Err(SynaptixError::Unsupported("rms_norm_residual_fused: scalar x"));
        }
        let h = self.dims()[xr - 1];
        if self.shape() != residual.shape() {
            return Err(SynaptixError::Unsupported("rms_norm_residual_fused: x/residual shape"));
        }
        if weight.rank() != 1 || weight.dims()[0] != h {
            return Err(SynaptixError::Unsupported("rms_norm_residual_fused: weight shape"));
        }
        if self.dtype() != residual.dtype() || self.dtype() != weight.dtype() {
            return Err(SynaptixError::Unsupported("rms_norm_residual_fused: dtype mismatch"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let r = if residual.is_contiguous() { residual.clone() } else { residual.contiguous()? };
        let w = if weight.is_contiguous() { weight.clone() } else { weight.contiguous()? };
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut hidden_st = backend.alloc_zeros(out_bytes, self.device())?;
        let mut y_st = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.rms_norm_residual(
            (&x.storage, &x.layout),
            (&r.storage, &r.layout),
            (&w.storage, &w.layout),
            (&mut hidden_st, &out_layout),
            (&mut y_st, &out_layout),
            eps,
            qwen_gain,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(hidden_st), out_layout.clone()),
            Tensor::from_parts(Arc::new(y_st), out_layout),
        ))
    }

    /// Квантует активацию (F16 `[.., K]`) в NVFP4 `(packed, scales)` — отдельно от
    /// GEMV. Квантуем общий `h` 1× → [`Self::linear_quant_prequant`] для каждой
    /// проекции из него (убирает дублирующие quantize-ядра). `Unsupported` на CPU.
    pub fn nvfp4_quantize_act(&self) -> Result<(Tensor, Tensor)> {
        if self.dtype() != DType::F16 {
            return Err(SynaptixError::Unsupported("nvfp4_quantize_act: x должен быть F16"));
        }
        let dims = self.dims();
        if dims.is_empty() {
            return Err(SynaptixError::Unsupported("nvfp4_quantize_act: scalar x"));
        }
        let k = dims[dims.len() - 1];
        if k == 0 || k % 16 != 0 {
            return Err(SynaptixError::Unsupported("nvfp4_quantize_act: K%16 != 0"));
        }
        let m = self.layout.numel() / k;
        // Размеры буферов NVFP4 — синхронно с kernels-cuda::nvfp4_scale_buffer_size.
        let packed_bytes = m * k / 2;
        let scales_bytes = (k.div_ceil(64) * 4) * (m.div_ceil(128) * 128);
        let x = self.contiguous_view()?;
        let backend = registry::backend_for(self.device())?;
        let mut packed_st = backend.alloc_zeros(packed_bytes, self.device())?;
        let mut scales_st = backend.alloc_zeros(scales_bytes, self.device())?;
        let packed_layout = Layout::contiguous(Shape::new(vec![packed_bytes]), DType::U8);
        let scales_layout = Layout::contiguous(Shape::new(vec![scales_bytes]), DType::U8);
        let stream = Stream::default_for(self.device())?;
        backend.nvfp4_quantize_act(
            (&x.storage, &x.layout),
            (&mut packed_st, &packed_layout),
            (&mut scales_st, &scales_layout),
            m,
            k,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(packed_st), packed_layout),
            Tensor::from_parts(Arc::new(scales_st), scales_layout),
        ))
    }

    pub fn silu_mul_quant_nvfp4(&self, inv_pre: f32) -> Result<(Tensor, Tensor)> {
        if !matches!(self.dtype(), DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("silu_mul_quant_nvfp4: dtype не F16/BF16"));
        }
        let dims = self.dims();
        if dims.is_empty() {
            return Err(SynaptixError::Unsupported("silu_mul_quant_nvfp4: scalar x"));
        }
        let k2 = dims[dims.len() - 1];
        if k2 == 0 || k2 % 32 != 0 {
            return Err(SynaptixError::Unsupported("silu_mul_quant_nvfp4: K%32 != 0"));
        }
        let k = k2 / 2;
        let m = self.layout.numel() / k2;
        let packed_bytes = m * k / 2;
        let scales_bytes = (k.div_ceil(64) * 4) * (m.div_ceil(128) * 128);
        let x = self.contiguous_view()?;
        let backend = registry::backend_for(self.device())?;
        let mut packed_st = backend.alloc_zeros(packed_bytes, self.device())?;
        let mut scales_st = backend.alloc_zeros(scales_bytes, self.device())?;
        let packed_layout = Layout::contiguous(Shape::new(vec![packed_bytes]), DType::U8);
        let scales_layout = Layout::contiguous(Shape::new(vec![scales_bytes]), DType::U8);
        let stream = Stream::default_for(self.device())?;
        backend.silu_mul_quant_nvfp4(
            (&x.storage, &x.layout),
            (&mut packed_st, &packed_layout),
            (&mut scales_st, &scales_layout),
            m,
            k,
            inv_pre,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(packed_st), packed_layout),
            Tensor::from_parts(Arc::new(scales_st), scales_layout),
        ))
    }

    /// Fused «adaLN-модуляция + NVFP4-квант» (эпилог нормы): за один launch
    /// `y = rms(self)·(1+scale)+shift` (бит-в-бит с decomposed-цепочкой
    /// rms→add_scalar→broadcast_mul→broadcast_add) и `(packed, scales) =`
    /// `nvfp4_quant(f16(y))` (бит-в-бит с [`Self::nvfp4_quantize_act`] от f16(y)).
    /// `self/scale/shift` `[.., K]` F16|BF16 одинаковых форм (по-токенная
    /// модуляция). Возвращает `(y, packed, scales)`. `Unsupported` вне CUDA.
    /// Fused gated-residual `self + y*g` (раунды как decomposed
    /// broadcast_mul→add → бит-в-бит): `g` либо same-shape (flat), либо
    /// broadcast-строка `[..,1,d]`. Err при неподходящих формах/бэкенде —
    /// вызывающий падает на decomposed.
    pub fn fused_gate_residual(&self, y: &Tensor, g: &Tensor) -> Result<Tensor> {
        let kind = if g.dims() == self.dims() {
            0u8
        } else if g.numel() == *self.dims().last().unwrap_or(&0) {
            1u8
        } else {
            return Err(SynaptixError::Unsupported("fused_gate_residual: форма g"));
        };
        let backend = registry::backend_for(self.device())?;
        let out_layout = Layout::contiguous(Shape::new(self.dims().to_vec()), self.dtype());
        let mut out_st = backend.alloc_uninit(self.dtype().bytes_for_numel(out_layout.numel()), self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.ternary_fused(
            kind,
            (&self.storage, &self.layout),
            (&y.storage, &y.layout),
            (&g.storage, &g.layout),
            (&mut out_st, &out_layout),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(out_st), out_layout))
    }

    /// Fused adaLN-модуляция готовой нормы `self*(1+s)+sh`, `s`/`sh` —
    /// broadcast-строки `[..,1,d]` (раунды как decomposed
    /// add_scalar→broadcast_mul→broadcast_add → бит-в-бит).
    pub fn fused_mod_row(&self, s: &Tensor, sh: &Tensor) -> Result<Tensor> {
        let kind = if s.dims() == self.dims() && sh.dims() == self.dims() {
            3u8
        } else {
            2u8
        };
        let backend = registry::backend_for(self.device())?;
        let out_layout = Layout::contiguous(Shape::new(self.dims().to_vec()), self.dtype());
        let mut out_st = backend.alloc_uninit(self.dtype().bytes_for_numel(out_layout.numel()), self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.ternary_fused(
            kind,
            (&self.storage, &self.layout),
            (&s.storage, &s.layout),
            (&sh.storage, &sh.layout),
            (&mut out_st, &out_layout),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(out_st), out_layout))
    }

    pub fn rms_mod_quant_nvfp4(
        &self,
        scale: &Tensor,
        shift: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dt = self.dtype();
        if !matches!(dt, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("rms_mod_quant: dtype не F16/BF16"));
        }
        if scale.dtype() != dt || shift.dtype() != dt {
            return Err(SynaptixError::Unsupported("rms_mod_quant: dtype scale/shift"));
        }
        if scale.dims() != self.dims() || shift.dims() != self.dims() {
            return Err(SynaptixError::Unsupported("rms_mod_quant: формы scale/shift"));
        }
        let dims = self.dims();
        if dims.is_empty() {
            return Err(SynaptixError::Unsupported("rms_mod_quant: scalar x"));
        }
        let k = dims[dims.len() - 1];
        if k == 0 || k % 16 != 0 {
            return Err(SynaptixError::Unsupported("rms_mod_quant: K%16 != 0"));
        }
        let m = self.layout.numel() / k;
        let packed_bytes = m * k / 2;
        let scales_bytes = (k.div_ceil(64) * 4) * (m.div_ceil(128) * 128);
        let x = self.contiguous_view()?;
        let sc = scale.contiguous_view()?;
        let sh = shift.contiguous_view()?;
        let backend = registry::backend_for(self.device())?;
        let y_layout = Layout::contiguous(Shape::new(dims.to_vec()), dt);
        let mut y_st = backend.alloc_zeros(dt.bytes_for_numel(y_layout.numel()), self.device())?;
        let mut packed_st = backend.alloc_zeros(packed_bytes, self.device())?;
        let mut scales_st = backend.alloc_zeros(scales_bytes, self.device())?;
        let packed_layout = Layout::contiguous(Shape::new(vec![packed_bytes]), DType::U8);
        let scales_layout = Layout::contiguous(Shape::new(vec![scales_bytes]), DType::U8);
        let stream = Stream::default_for(self.device())?;
        backend.rms_mod_quant_nvfp4(
            (&x.storage, &x.layout),
            (&sc.storage, &sc.layout),
            (&sh.storage, &sh.layout),
            (&mut y_st, &y_layout),
            &mut packed_st,
            &mut scales_st,
            m,
            k,
            eps,
            0,
            1,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(y_st), y_layout),
            Tensor::from_parts(Arc::new(packed_st), packed_layout),
            Tensor::from_parts(Arc::new(scales_st), scales_layout),
        ))
    }

    /// RMS+вес-вариант (LLM prefill, без модуляции): `y = rms(self)·w`
    /// (`qwen` → `·(1+w)`) + NVFP4-квант f16(y). Бит-в-бит с
    /// rms_norm_fused(w, eps, qwen) → quantize_act.
    pub fn rms_quant_nvfp4(
        &self,
        w: &Tensor,
        eps: f32,
        qwen: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dt = self.dtype();
        if !matches!(dt, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("rms_quant: dtype не F16/BF16"));
        }
        if w.dtype() != dt {
            return Err(SynaptixError::Unsupported("rms_quant: dtype w"));
        }
        let dims = self.dims();
        if dims.is_empty() {
            return Err(SynaptixError::Unsupported("rms_quant: scalar x"));
        }
        let k = dims[dims.len() - 1];
        if k == 0 || k % 16 != 0 || w.layout.numel() != k {
            return Err(SynaptixError::Unsupported("rms_quant: K%16 или форма w"));
        }
        let m = self.layout.numel() / k;
        let packed_bytes = m * k / 2;
        let scales_bytes = (k.div_ceil(64) * 4) * (m.div_ceil(128) * 128);
        let x = self.contiguous_view()?;
        let wv = w.contiguous_view()?;
        let backend = registry::backend_for(self.device())?;
        let y_layout = Layout::contiguous(Shape::new(dims.to_vec()), dt);
        let mut y_st = backend.alloc_zeros(dt.bytes_for_numel(y_layout.numel()), self.device())?;
        let mut packed_st = backend.alloc_zeros(packed_bytes, self.device())?;
        let mut scales_st = backend.alloc_zeros(scales_bytes, self.device())?;
        let packed_layout = Layout::contiguous(Shape::new(vec![packed_bytes]), DType::U8);
        let scales_layout = Layout::contiguous(Shape::new(vec![scales_bytes]), DType::U8);
        let stream = Stream::default_for(self.device())?;
        backend.rms_mod_quant_nvfp4(
            (&x.storage, &x.layout),
            (&wv.storage, &wv.layout),
            (&wv.storage, &wv.layout),
            (&mut y_st, &y_layout),
            &mut packed_st,
            &mut scales_st,
            m,
            k,
            eps,
            if qwen { 3 } else { 2 },
            1,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(y_st), y_layout),
            Tensor::from_parts(Arc::new(packed_st), packed_layout),
            Tensor::from_parts(Arc::new(scales_st), scales_layout),
        ))
    }

    /// LN-вариант [`Self::rms_mod_quant_nvfp4`] (FLUX adaLN): `y = LN(self)·(1+scale)+shift`
    /// + NVFP4-квант f16(y). `scale/shift` — per-batch векторы `[B, K]` (broadcast по
    /// последовательности), бит-в-бит с layer_norm(ones)→modulate→quantize.
    pub fn ln_mod_quant_nvfp4(
        &self,
        scale: &Tensor,
        shift: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dt = self.dtype();
        if !matches!(dt, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("ln_mod_quant: dtype не F16/BF16"));
        }
        if scale.dtype() != dt || shift.dtype() != dt {
            return Err(SynaptixError::Unsupported("ln_mod_quant: dtype scale/shift"));
        }
        let dims = self.dims();
        if dims.len() < 2 {
            return Err(SynaptixError::Unsupported("ln_mod_quant: rank < 2"));
        }
        let k = dims[dims.len() - 1];
        if k == 0 || k % 16 != 0 {
            return Err(SynaptixError::Unsupported("ln_mod_quant: K%16 != 0"));
        }
        let m = self.layout.numel() / k;
        let b = scale.layout.numel() / k;
        if b == 0 || m % b != 0 || shift.layout.numel() != scale.layout.numel() {
            return Err(SynaptixError::Unsupported("ln_mod_quant: формы scale/shift"));
        }
        let mod_div = m / b;
        let packed_bytes = m * k / 2;
        let scales_bytes = (k.div_ceil(64) * 4) * (m.div_ceil(128) * 128);
        let x = self.contiguous_view()?;
        let sc = scale.contiguous_view()?;
        let sh = shift.contiguous_view()?;
        let backend = registry::backend_for(self.device())?;
        let y_layout = Layout::contiguous(Shape::new(dims.to_vec()), dt);
        let mut y_st = backend.alloc_zeros(dt.bytes_for_numel(y_layout.numel()), self.device())?;
        let mut packed_st = backend.alloc_zeros(packed_bytes, self.device())?;
        let mut scales_st = backend.alloc_zeros(scales_bytes, self.device())?;
        let packed_layout = Layout::contiguous(Shape::new(vec![packed_bytes]), DType::U8);
        let scales_layout = Layout::contiguous(Shape::new(vec![scales_bytes]), DType::U8);
        let stream = Stream::default_for(self.device())?;
        backend.rms_mod_quant_nvfp4(
            (&x.storage, &x.layout),
            (&sc.storage, &sc.layout),
            (&sh.storage, &sh.layout),
            (&mut y_st, &y_layout),
            &mut packed_st,
            &mut scales_st,
            m,
            k,
            eps,
            1,
            mod_div,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(y_st), y_layout),
            Tensor::from_parts(Arc::new(packed_st), packed_layout),
            Tensor::from_parts(Arc::new(scales_st), scales_layout),
        ))
    }

    /// Квантует активацию (F16 `[.., K]`) в MXFP8 `(packed [m·k], scales natural
    /// [m·k/32])` — для шаринга между проекциями с MXFP8-весом
    /// ([`Self::linear_quant_prequant`]). `Unsupported` на CPU.
    pub fn mxfp8_quantize_act(&self) -> Result<(Tensor, Tensor)> {
        if self.dtype() != DType::F16 {
            return Err(SynaptixError::Unsupported("mxfp8_quantize_act: x должен быть F16"));
        }
        let dims = self.dims();
        if dims.is_empty() {
            return Err(SynaptixError::Unsupported("mxfp8_quantize_act: scalar x"));
        }
        let k = dims[dims.len() - 1];
        if k == 0 || k % 32 != 0 {
            return Err(SynaptixError::Unsupported("mxfp8_quantize_act: K%32 != 0"));
        }
        let m = self.layout.numel() / k;
        let packed_bytes = m * k;
        let scales_bytes = m * (k / 32);
        let x = self.contiguous_view()?;
        let backend = registry::backend_for(self.device())?;
        let mut packed_st = backend.alloc_zeros(packed_bytes, self.device())?;
        let mut scales_st = backend.alloc_zeros(scales_bytes, self.device())?;
        let packed_layout = Layout::contiguous(Shape::new(vec![packed_bytes]), DType::U8);
        let scales_layout = Layout::contiguous(Shape::new(vec![scales_bytes]), DType::U8);
        let stream = Stream::default_for(self.device())?;
        backend.mxfp8_quantize_act(
            (&x.storage, &x.layout),
            (&mut packed_st, &packed_layout),
            (&mut scales_st, &scales_layout),
            m,
            k,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(packed_st), packed_layout),
            Tensor::from_parts(Arc::new(scales_st), scales_layout),
        ))
    }

    /// MXFP8-вариант [`Self::rms_mod_quant_nvfp4`]: та же норма+модуляция
    /// бит-в-бит, эпилог = MXFP8-квант (бит-в-бит с [`Self::mxfp8_quantize_act`]
    /// от f16(y)). Возвращает `(y, packed [m·k], scales natural [m·k/32])`.
    pub fn rms_mod_quant_mxfp8(
        &self,
        scale: &Tensor,
        shift: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dt = self.dtype();
        if !matches!(dt, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("rms_mod_quant_mxfp8: dtype не F16/BF16"));
        }
        if scale.dtype() != dt || shift.dtype() != dt {
            return Err(SynaptixError::Unsupported("rms_mod_quant_mxfp8: dtype scale/shift"));
        }
        if scale.dims() != self.dims() || shift.dims() != self.dims() {
            return Err(SynaptixError::Unsupported("rms_mod_quant_mxfp8: формы scale/shift"));
        }
        let dims = self.dims();
        if dims.is_empty() {
            return Err(SynaptixError::Unsupported("rms_mod_quant_mxfp8: scalar x"));
        }
        let k = dims[dims.len() - 1];
        if k == 0 || k % 32 != 0 {
            return Err(SynaptixError::Unsupported("rms_mod_quant_mxfp8: K%32 != 0"));
        }
        let m = self.layout.numel() / k;
        let packed_bytes = m * k;
        let scales_bytes = m * (k / 32);
        let x = self.contiguous_view()?;
        let sc = scale.contiguous_view()?;
        let sh = shift.contiguous_view()?;
        let backend = registry::backend_for(self.device())?;
        let y_layout = Layout::contiguous(Shape::new(dims.to_vec()), dt);
        let mut y_st = backend.alloc_zeros(dt.bytes_for_numel(y_layout.numel()), self.device())?;
        let mut packed_st = backend.alloc_zeros(packed_bytes, self.device())?;
        let mut scales_st = backend.alloc_zeros(scales_bytes, self.device())?;
        let packed_layout = Layout::contiguous(Shape::new(vec![packed_bytes]), DType::U8);
        let scales_layout = Layout::contiguous(Shape::new(vec![scales_bytes]), DType::U8);
        let stream = Stream::default_for(self.device())?;
        backend.rms_mod_quant_mxfp8(
            (&x.storage, &x.layout),
            (&sc.storage, &sc.layout),
            (&sh.storage, &sh.layout),
            (&mut y_st, &y_layout),
            &mut packed_st,
            &mut scales_st,
            m,
            k,
            eps,
            0,
            1,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(y_st), y_layout),
            Tensor::from_parts(Arc::new(packed_st), packed_layout),
            Tensor::from_parts(Arc::new(scales_st), scales_layout),
        ))
    }

    /// MXFP8-вариант [`Self::rms_quant_nvfp4`] (LLM prefill): `y = rms(self)·w`
    /// (`qwen` → `·(1+w)`) + MXFP8-квант f16(y).
    pub fn rms_quant_mxfp8(
        &self,
        w: &Tensor,
        eps: f32,
        qwen: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dt = self.dtype();
        if !matches!(dt, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("rms_quant_mxfp8: dtype не F16/BF16"));
        }
        if w.dtype() != dt {
            return Err(SynaptixError::Unsupported("rms_quant_mxfp8: dtype w"));
        }
        let dims = self.dims();
        if dims.is_empty() {
            return Err(SynaptixError::Unsupported("rms_quant_mxfp8: scalar x"));
        }
        let k = dims[dims.len() - 1];
        if k == 0 || k % 32 != 0 || w.layout.numel() != k {
            return Err(SynaptixError::Unsupported("rms_quant_mxfp8: K%32 или форма w"));
        }
        let m = self.layout.numel() / k;
        let packed_bytes = m * k;
        let scales_bytes = m * (k / 32);
        let x = self.contiguous_view()?;
        let wv = w.contiguous_view()?;
        let backend = registry::backend_for(self.device())?;
        let y_layout = Layout::contiguous(Shape::new(dims.to_vec()), dt);
        let mut y_st = backend.alloc_zeros(dt.bytes_for_numel(y_layout.numel()), self.device())?;
        let mut packed_st = backend.alloc_zeros(packed_bytes, self.device())?;
        let mut scales_st = backend.alloc_zeros(scales_bytes, self.device())?;
        let packed_layout = Layout::contiguous(Shape::new(vec![packed_bytes]), DType::U8);
        let scales_layout = Layout::contiguous(Shape::new(vec![scales_bytes]), DType::U8);
        let stream = Stream::default_for(self.device())?;
        backend.rms_mod_quant_mxfp8(
            (&x.storage, &x.layout),
            (&wv.storage, &wv.layout),
            (&wv.storage, &wv.layout),
            (&mut y_st, &y_layout),
            &mut packed_st,
            &mut scales_st,
            m,
            k,
            eps,
            if qwen { 3 } else { 2 },
            1,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(y_st), y_layout),
            Tensor::from_parts(Arc::new(packed_st), packed_layout),
            Tensor::from_parts(Arc::new(scales_st), scales_layout),
        ))
    }

    /// MXFP8-вариант [`Self::ln_mod_quant_nvfp4`] (FLUX adaLN): `y = LN(self)·(1+scale)+shift`
    /// + MXFP8-квант f16(y). `scale/shift` — per-batch векторы `[B, K]`.
    pub fn ln_mod_quant_mxfp8(
        &self,
        scale: &Tensor,
        shift: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dt = self.dtype();
        if !matches!(dt, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("ln_mod_quant_mxfp8: dtype не F16/BF16"));
        }
        if scale.dtype() != dt || shift.dtype() != dt {
            return Err(SynaptixError::Unsupported("ln_mod_quant_mxfp8: dtype scale/shift"));
        }
        let dims = self.dims();
        if dims.len() < 2 {
            return Err(SynaptixError::Unsupported("ln_mod_quant_mxfp8: rank < 2"));
        }
        let k = dims[dims.len() - 1];
        if k == 0 || k % 32 != 0 {
            return Err(SynaptixError::Unsupported("ln_mod_quant_mxfp8: K%32 != 0"));
        }
        let m = self.layout.numel() / k;
        let b = scale.layout.numel() / k;
        if b == 0 || m % b != 0 || shift.layout.numel() != scale.layout.numel() {
            return Err(SynaptixError::Unsupported("ln_mod_quant_mxfp8: формы scale/shift"));
        }
        let mod_div = m / b;
        let packed_bytes = m * k;
        let scales_bytes = m * (k / 32);
        let x = self.contiguous_view()?;
        let sc = scale.contiguous_view()?;
        let sh = shift.contiguous_view()?;
        let backend = registry::backend_for(self.device())?;
        let y_layout = Layout::contiguous(Shape::new(dims.to_vec()), dt);
        let mut y_st = backend.alloc_zeros(dt.bytes_for_numel(y_layout.numel()), self.device())?;
        let mut packed_st = backend.alloc_zeros(packed_bytes, self.device())?;
        let mut scales_st = backend.alloc_zeros(scales_bytes, self.device())?;
        let packed_layout = Layout::contiguous(Shape::new(vec![packed_bytes]), DType::U8);
        let scales_layout = Layout::contiguous(Shape::new(vec![scales_bytes]), DType::U8);
        let stream = Stream::default_for(self.device())?;
        backend.rms_mod_quant_mxfp8(
            (&x.storage, &x.layout),
            (&sc.storage, &sc.layout),
            (&sh.storage, &sh.layout),
            (&mut y_st, &y_layout),
            &mut packed_st,
            &mut scales_st,
            m,
            k,
            eps,
            1,
            mod_div,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(y_st), y_layout),
            Tensor::from_parts(Arc::new(packed_st), packed_layout),
            Tensor::from_parts(Arc::new(scales_st), scales_layout),
        ))
    }

    /// GEMM/GEMV из УЖЕ квантованной активации (`self` = packed от
    /// [`Self::nvfp4_quantize_act`] | [`Self::mxfp8_quantize_act`] — формат по
    /// `w.dtype()`). `m` — число строк (decode = 1). Возвращает
    /// `[m, w.n()]` F16. `Unsupported` если backend не поддержал prequant-путь.
    pub fn linear_quant_prequant(
        &self,
        scales: &Tensor,
        w: &QuantWeight,
        m: usize,
        out_dt: DType,
    ) -> Result<Tensor> {
        if !matches!(out_dt, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("linear_quant_prequant: out_dt не F16/BF16"));
        }
        let backend = registry::backend_for(self.device())?;
        let out_layout = Layout::contiguous(Shape::new(vec![m, w.n()]), out_dt);
        let out_bytes = out_dt.bytes_for_numel(out_layout.numel());
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.linear_quant_prequant(
            &self.storage,
            &scales.storage,
            w,
            (&mut storage, &out_layout),
            m,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fused GroupNorm через backend (CUDA — один launch). `self[B,C,*spatial]`,
    /// `weight`/`bias` опц. `[C]`. Возвращает `Unsupported`, если backend не умеет
    /// (CPU) — `synaptix-ops::group_norm` тогда падает в decomposed путь.
    pub fn group_norm_fused(
        &self,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        num_groups: usize,
        eps: f32,
        silu: bool,
    ) -> Result<Self> {
        self.group_norm_fused_layout(weight, bias, num_groups, eps, silu, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn group_norm_fused_layout(
        &self,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        num_groups: usize,
        eps: f32,
        silu: bool,
        nhwc: bool,
    ) -> Result<Self> {
        if self.rank() < 2 {
            return Err(SynaptixError::Unsupported("group_norm_fused: rank < 2"));
        }
        let c = if nhwc { self.dims()[self.rank() - 1] } else { self.dims()[1] };
        if num_groups == 0 || c % num_groups != 0 {
            return Err(SynaptixError::Unsupported("group_norm_fused: c % num_groups"));
        }
        let chk = |t: &Tensor| -> Result<Tensor> {
            if t.rank() != 1 || t.dims()[0] != c || t.dtype() != self.dtype() {
                return Err(SynaptixError::Unsupported("group_norm_fused: affine shape/dtype"));
            }
            if t.is_contiguous() { Ok(t.clone()) } else { t.contiguous() }
        };
        let w = weight.map(chk).transpose()?;
        let b = bias.map(chk).transpose()?;
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.group_norm(
            (&x.storage, &x.layout),
            w.as_ref().map(|t| (&*t.storage, &t.layout)),
            b.as_ref().map(|t| (&*t.storage, &t.layout)),
            (&mut storage, &out_layout),
            num_groups,
            eps,
            silu,
            nhwc,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fused conv2d-эпилог: `self[B*H*W, C]` (NHWC-flat) + опц. `bias[C]` →
    /// `[B, C, H, W]` (NCHW) одним проходом. Заменяет `broadcast_add(bias)` +
    /// `permute([0,3,1,2]).contiguous()` (2 прохода → 1).
    pub fn conv_epilogue(
        &self,
        bias: Option<&Tensor>,
        residual: Option<&Tensor>,
        temb_bc: Option<&Tensor>,
        b: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<Self> {
        if self.rank() != 2 || self.dims()[0] != b * h * w || self.dims()[1] != c {
            return Err(SynaptixError::Unsupported("conv_epilogue: self shape"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let bs = match bias {
            Some(bt) => {
                if bt.rank() != 1 || bt.dims()[0] != c || bt.dtype() != self.dtype() {
                    return Err(SynaptixError::Unsupported("conv_epilogue: bias shape/dtype"));
                }
                Some(if bt.is_contiguous() { bt.clone() } else { bt.contiguous()? })
            }
            None => None,
        };
        let res = match residual {
            Some(rt) => {
                if rt.rank() != 4
                    || rt.dims() != [b, c, h, w]
                    || rt.dtype() != self.dtype()
                {
                    return Err(SynaptixError::Unsupported("conv_epilogue: residual shape/dtype"));
                }
                Some(if rt.is_contiguous() { rt.clone() } else { rt.contiguous()? })
            }
            None => None,
        };
        let temb = match temb_bc {
            Some(tt) => {
                if tt.rank() != 2 || tt.dims() != [b, c] || tt.dtype() != self.dtype() {
                    return Err(SynaptixError::Unsupported("conv_epilogue: temb shape/dtype"));
                }
                Some(if tt.is_contiguous() { tt.clone() } else { tt.contiguous()? })
            }
            None => None,
        };
        let out_layout = Layout::contiguous(Shape::new(vec![b, c, h, w]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.conv_epilogue(
            (&x.storage, &x.layout),
            bs.as_ref().map(|t| (&*t.storage, &t.layout)),
            res.as_ref().map(|t| (&*t.storage, &t.layout)),
            temb.as_ref().map(|t| (&*t.storage, &t.layout)),
            (&mut storage, &out_layout),
            b, c, h, w,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fused GEGLU split: `self[.., 2*I]` → `[.., I]`, `out[t,i] = self[t,i] *
    /// gelu_exact(self[t, I+i])`. `Unsupported` на CPU (caller — narrow+gelu+mul).
    pub fn geglu_split(&self) -> Result<Self> {
        let r = self.rank();
        if r == 0 {
            return Err(SynaptixError::Unsupported("geglu_split: scalar"));
        }
        let last = self.dims()[r - 1];
        if last % 2 != 0 {
            return Err(SynaptixError::Unsupported("geglu_split: last dim odd"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let mut out_dims = self.dims().to_vec();
        out_dims[r - 1] = last / 2;
        let out_layout = Layout::contiguous(Shape::new(out_dims), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.geglu_split((&x.storage, &x.layout), (&mut storage, &out_layout), &stream)?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fused Snake-активация (Oobleck VAE): `out = self + sin(exp(alpha)*self)^2
    /// / (exp(beta)+eps)` по канальной оси (dim 1). `alpha`/`beta` — `[1,C,1]`
    /// или `[C]`. Предвычисляет per-channel `a=exp(alpha)`, `binv=1/(exp(beta)+eps)`
    /// (крошечные `[C]`-ops), затем один большой проход. `Unsupported` (CPU) →
    /// caller падает в decomposed путь.
    pub fn snake(&self, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Self> {
        if self.rank() < 2 {
            return Err(SynaptixError::Unsupported("snake: rank < 2"));
        }
        let c = self.dims()[1];
        let t_len: usize = self.dims()[2..].iter().product::<usize>().max(1);
        let a = alpha.to_dtype(DType::F32)?.exp()?.flatten_all()?.contiguous()?;
        let binv = beta
            .to_dtype(DType::F32)?
            .exp()?
            .affine(1.0, eps)?
            .recip()?
            .flatten_all()?
            .contiguous()?;
        if a.dims() != [c] || binv.dims() != [c] {
            return Err(SynaptixError::Unsupported("snake: alpha/beta shape != [C]"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(self.dims().to_vec()), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.snake(
            (&x.storage, &x.layout),
            (&a.storage, &a.layout),
            (&binv.storage, &binv.layout),
            (&mut storage, &out_layout),
            c,
            t_len,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// nearest-2x upsample через backend: `self[B,C,H,W]` → `[B,C,2H,2W]`,
    /// `out[b,c,ho,wo]=self[b,c,ho/2,wo/2]`. `Unsupported` на CPU (caller падает
    /// в cat-based reshape).
    pub fn upsample_nearest2x(&self) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("upsample_nearest2x: rank != 4"));
        }
        let d = self.dims();
        let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout =
            Layout::contiguous(Shape::new(vec![b, c, h * 2, w * 2]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.upsample_nearest2x(
            (&x.storage, &x.layout),
            (&mut storage, &out_layout),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// im2col через backend: `self[B,C_in,H,W]` → `col[M,K]`, `M=B*h_out*w_out`,
    /// `K=C_in*kh*kw`. Питает conv2d-через-GEMM. `Unsupported` на CPU.
    pub fn im2col(
        &self,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        h_out: usize,
        w_out: usize,
        m_offset: usize,
        m_count: usize,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("im2col: rank != 4"));
        }
        let c_in = self.dims()[1];
        let k = c_in * kh * kw;
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![m_count, k]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.im2col(
            (&x.storage, &x.layout),
            (&mut storage, &out_layout),
            kh,
            kw,
            h_out,
            w_out,
            stride,
            padding,
            m_offset as u64,
            m_count as u64,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fast 4D permute NCHW [B,C,H,W] → NHWC [B,H,W,C] через shmem-tile-ядро.
    /// Fallback: generic permute([0,2,3,1]).contiguous().
    pub fn nchw_to_nhwc(&self) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("nchw_to_nhwc: rank != 4"));
        }
        let d = self.dims();
        let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout =
            Layout::contiguous(Shape::new(vec![b, h, w, c]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        match backend.nchw_to_nhwc(
            (&x.storage, &x.layout),
            (&mut storage, &out_layout),
            &stream,
        ) {
            Ok(()) => Ok(Tensor::from_parts(Arc::new(storage), out_layout)),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                // CPU / неподдержка → generic.
                x.permute(vec![0, 2, 3, 1])?.contiguous()
            }
            Err(e) => Err(e),
        }
    }

    /// Обратный fast-permute NHWC `[B,H,W,C]` → NCHW `[B,C,H,W]` (shmem-tile,
    /// fallback generic permute). Парный к [`Tensor::nchw_to_nhwc`].
    pub fn nhwc_to_nchw(&self) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("nhwc_to_nchw: rank != 4"));
        }
        let d = self.dims();
        let (b, h, w, c) = (d[0], d[1], d[2], d[3]);
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![b, c, h, w]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        match backend.nhwc_to_nchw(
            (&x.storage, &x.layout),
            (&mut storage, &out_layout),
            &stream,
        ) {
            Ok(()) => Ok(Tensor::from_parts(Arc::new(storage), out_layout)),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                x.permute(vec![0, 3, 1, 2])?.contiguous()
            }
            Err(e) => Err(e),
        }
    }

    /// CUTLASS Implicit-GEMM conv2d (cuDNN-стиль, без im2col K-blowup).
    /// Транспонирует input NCHW→NHWC и filter [Cout,Cin,Kh,Kw]→[Cout,Kh,Kw,Cin],
    /// зовёт backend.conv2d_implicit_nhwc, возвращает выход как `[B*Hout*Wout, Cout]`
    /// (NHWC-flat, готов для conv_epilogue). `Unsupported` на CPU / при Cin%8≠0
    /// (caller падает в im2col-путь).
    #[allow(clippy::too_many_arguments)]
    fn try_conv2d_nhwc(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        residual: Option<&Tensor>,
        temb: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        out_h: usize,
        out_w: usize,
        in_is_nhwc: bool,
        out_nhwc: bool,
    ) -> Result<Self> {
        if !matches!(self.dtype(), DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("try_conv2d_nhwc: dtype"));
        }
        let b = self.dims()[0];
        let c_in = if in_is_nhwc { self.dims()[3] } else { self.dims()[1] };
        let c_out = weight.dims()[0];
        if c_in % 8 != 0 || c_out % 8 != 0 {
            return Err(SynaptixError::Unsupported("try_conv2d_nhwc: Cin/Cout %8"));
        }
        let input_nhwc = if in_is_nhwc {
            if self.is_contiguous() { self.clone() } else { self.contiguous()? }
        } else {
            self.nchw_to_nhwc()?
        };
        let filter_krsc = cached_filter_krsc(weight)?;
        let out_dims = if out_nhwc {
            vec![b, out_h, out_w, c_out]
        } else {
            vec![b, c_out, out_h, out_w]
        };
        let out_layout = Layout::contiguous(Shape::new(out_dims), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        let bias_arg = bias.map(|t| (&*t.storage, &t.layout));
        let res_c = match residual {
            Some(r) if r.is_contiguous() => Some(r.clone()),
            Some(r) => Some(r.contiguous()?),
            None => None,
        };
        let res_arg = res_c.as_ref().map(|r| (&*r.storage, &r.layout));
        let temb_arg = temb.map(|t| (&*t.storage, &t.layout));
        backend.conv2d_implicit_nhwc(
            (&input_nhwc.storage, &input_nhwc.layout),
            (&filter_krsc.storage, &filter_krsc.layout),
            bias_arg,
            res_arg,
            temb_arg,
            (&mut storage, &out_layout),
            out_nhwc,
            stride,
            padding,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// NHWC-throughout conv: вход `[B,H,W,Cin]`, выход `[B,P,Q,Cout]`. Опц. fused
    /// bias[Cout] + residual[B,P,Q,Cout] (NHWC) + temb[B,Cout]. `Unsupported` →
    /// caller падает в NCHW-путь.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_nhwc_io(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        residual: Option<&Tensor>,
        temb: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        out_h: usize,
        out_w: usize,
    ) -> Result<Self> {
        self.try_conv2d_nhwc(
            weight, bias, residual, temb, stride, padding, out_h, out_w, true, true,
        )
    }

    /// conv2d через im2col + GEMM: `col[M,K] @ Wᵀ[K,C_out]` (быстрый cutlass-GEMM,
    /// на порядки быстрее direct-conv на больших каналах). `Unsupported`, если
    /// backend не умеет im2col (CPU) или dense-GEMM (нет cutlass) — caller падает
    /// в direct-conv путь.
    fn conv2d_im2col(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        out_h: usize,
        out_w: usize,
    ) -> Result<Self> {
        let (b, c_in) = (self.dims()[0], self.dims()[1]);
        let (c_out, kh, kw) = (weight.dims()[0], weight.dims()[2], weight.dims()[3]);
        // Implicit-GEMM первым приоритетом: NCHW-выход + fused bias эпилог, без im2col.
        match self.try_conv2d_nhwc(weight, bias, None, None, stride, padding, out_h, out_w, false, false) {
            Ok(t) => return Ok(t),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
        let out2d = {
                // Fallback: im2col + GEMM с row-tiling.
                let k = c_in * kh * kw;
                let m = b * out_h * out_w;
                let wt2 =
                    weight.reshape(vec![c_out, k])?.transpose(0, 1)?.contiguous()?;
                const MAX_COL_BYTES: usize = 1024 * 1024 * 1024;
                let esz = (self.dtype().size_in_bits() / 8) as usize;
                let max_rows = (MAX_COL_BYTES / (k * esz).max(1)).max(1);
                if m <= max_rows {
                    self.im2col(kh, kw, stride, padding, out_h, out_w, 0, m)?.matmul(&wt2)?
                } else {
                    let mut parts: Vec<Tensor> = Vec::with_capacity(m.div_ceil(max_rows));
                    let mut m0 = 0;
                    while m0 < m {
                        let mc = max_rows.min(m - m0);
                        let col = self.im2col(kh, kw, stride, padding, out_h, out_w, m0, mc)?;
                        parts.push(col.matmul(&wt2)?);
                        m0 += mc;
                    }
                    let refs: Vec<&Tensor> = parts.iter().collect();
                    Tensor::cat(&refs, 0)?
                }
        };
        // Fallback conv-эпилог (im2col-путь): bias-add + NHWC→NCHW transpose.
        match out2d.conv_epilogue(bias, None, None, b, c_out, out_h, out_w) {
            Ok(out) => return Ok(out),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
        let out2d = match bias {
            Some(bt) => out2d.broadcast_add(&bt.reshape(vec![1, c_out])?)?,
            None => out2d,
        };
        out2d
            .reshape(vec![b, out_h, out_w, c_out])?
            .permute(vec![0, 3, 1, 2])?
            .contiguous()
    }

    /// Conv2d + temb-broadcast в одном fused-эпилоге: `out = conv2d(self,w,b)
    /// + temb[B,C,None,None]`. Для resnet conv1 (заменяет conv + broadcast_add).
    pub fn conv2d_temb(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        temb: &Tensor,
    ) -> Result<Self> {
        if self.rank() != 4 || weight.rank() != 4 {
            return Err(SynaptixError::Unsupported("conv2d_temb: rank != 4"));
        }
        let (b, c_in, h, w) = (self.dims()[0], self.dims()[1], self.dims()[2], self.dims()[3]);
        let (c_out, c_in_w, kh, kw) =
            (weight.dims()[0], weight.dims()[1], weight.dims()[2], weight.dims()[3]);
        if c_in != c_in_w || self.dtype() != weight.dtype() {
            return Err(SynaptixError::Unsupported("conv2d_temb: shape/dtype mismatch"));
        }
        let (sh, sw) = (stride.0.max(1), stride.1.max(1));
        let (ph, pw) = padding;
        if h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(SynaptixError::Unsupported("conv2d_temb: input too small"));
        }
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;
        if temb.rank() != 2 || temb.dims() != [b, c_out] || temb.dtype() != self.dtype() {
            return Err(SynaptixError::Unsupported("conv2d_temb: temb shape/dtype"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let wt = if weight.is_contiguous() { weight.clone() } else { weight.contiguous()? };
        let bs = match bias {
            Some(bt) => {
                if bt.rank() != 1 || bt.dims()[0] != c_out || bt.dtype() != self.dtype() {
                    return Err(SynaptixError::Unsupported("conv2d_temb: bias shape/dtype"));
                }
                Some(if bt.is_contiguous() { bt.clone() } else { bt.contiguous()? })
            }
            None => None,
        };
        let tb = if temb.is_contiguous() { temb.clone() } else { temb.contiguous()? };
        // Implicit-GEMM первым приоритетом: NCHW-выход + fused bias+temb эпилог.
        match x.try_conv2d_nhwc(&wt, bs.as_ref(), None, Some(&tb), (sh, sw), (ph, pw), out_h, out_w, false, false) {
            Ok(t) => return Ok(t),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
        let via_im2col = (|| -> Result<Self> {
            let k = c_in * kh * kw;
            let m = b * out_h * out_w;
            let wt2 = wt.reshape(vec![c_out, k])?.transpose(0, 1)?.contiguous()?;
            const MAX_COL_BYTES: usize = 1024 * 1024 * 1024;
            let esz = (self.dtype().size_in_bits() / 8) as usize;
            let max_rows = (MAX_COL_BYTES / (k * esz).max(1)).max(1);
            let out2d = if m <= max_rows {
                x.im2col(kh, kw, (sh, sw), (ph, pw), out_h, out_w, 0, m)?.matmul(&wt2)?
            } else {
                let mut parts: Vec<Tensor> = Vec::with_capacity(m.div_ceil(max_rows));
                let mut m0 = 0;
                while m0 < m {
                    let mc = max_rows.min(m - m0);
                    let col = x.im2col(kh, kw, (sh, sw), (ph, pw), out_h, out_w, m0, mc)?;
                    parts.push(col.matmul(&wt2)?);
                    m0 += mc;
                }
                let refs: Vec<&Tensor> = parts.iter().collect();
                Tensor::cat(&refs, 0)?
            };
            out2d.conv_epilogue(bs.as_ref(), None, Some(&tb), b, c_out, out_h, out_w)
        })();
        match via_im2col {
            Ok(out) => return Ok(out),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
        // Fallback (CPU/нет im2col): conv + broadcast_add (temb [B,C,1,1] → broadcast).
        let conv_out = self.conv2d(weight, bias, stride, padding)?;
        let t4 = tb.reshape(vec![b, c_out, 1, 1])?;
        conv_out.broadcast_add(&t4)
    }

    /// Conv2d + residual-add в одном fused-эпилоге: `out = conv2d(self,w,b) +
    /// residual`. Используется в ResNet (заменяет `conv(x).add(res)` =
    /// extra binary pass). На неподдержке падаем в `conv2d + add`.
    pub fn conv2d_add(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        residual: &Tensor,
    ) -> Result<Self> {
        // Валидация (как в conv2d).
        if self.rank() != 4 || weight.rank() != 4 {
            return Err(SynaptixError::Unsupported("conv2d_add: input/weight rank != 4"));
        }
        let (b, c_in, h, w) = (self.dims()[0], self.dims()[1], self.dims()[2], self.dims()[3]);
        let (c_out, c_in_w, kh, kw) =
            (weight.dims()[0], weight.dims()[1], weight.dims()[2], weight.dims()[3]);
        if c_in != c_in_w || self.dtype() != weight.dtype() {
            return Err(SynaptixError::Unsupported("conv2d_add: shape/dtype mismatch"));
        }
        let (sh, sw) = (stride.0.max(1), stride.1.max(1));
        let (ph, pw) = padding;
        if h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(SynaptixError::Unsupported("conv2d_add: input too small"));
        }
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;
        if residual.dims() != [b, c_out, out_h, out_w] || residual.dtype() != self.dtype() {
            return Err(SynaptixError::Unsupported("conv2d_add: residual shape/dtype"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let wt = if weight.is_contiguous() { weight.clone() } else { weight.contiguous()? };
        let bs = match bias {
            Some(bt) => {
                if bt.rank() != 1 || bt.dims()[0] != c_out || bt.dtype() != self.dtype() {
                    return Err(SynaptixError::Unsupported("conv2d_add: bias shape/dtype"));
                }
                Some(if bt.is_contiguous() { bt.clone() } else { bt.contiguous()? })
            }
            None => None,
        };
        let res = if residual.is_contiguous() { residual.clone() } else { residual.contiguous()? };

        // Implicit-GEMM первым приоритетом: NCHW-выход + fused bias+residual эпилог.
        match x.try_conv2d_nhwc(&wt, bs.as_ref(), Some(&res), None, (sh, sw), (ph, pw), out_h, out_w, false, false) {
            Ok(t) => return Ok(t),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
        let via_im2col = (|| -> Result<Self> {
            let k = c_in * kh * kw;
            let m = b * out_h * out_w;
            let wt2 = wt.reshape(vec![c_out, k])?.transpose(0, 1)?.contiguous()?;
            const MAX_COL_BYTES: usize = 1024 * 1024 * 1024;
            let esz = (self.dtype().size_in_bits() / 8) as usize;
            let max_rows = (MAX_COL_BYTES / (k * esz).max(1)).max(1);
            let out2d = if m <= max_rows {
                x.im2col(kh, kw, (sh, sw), (ph, pw), out_h, out_w, 0, m)?.matmul(&wt2)?
            } else {
                let mut parts: Vec<Tensor> = Vec::with_capacity(m.div_ceil(max_rows));
                let mut m0 = 0;
                while m0 < m {
                    let mc = max_rows.min(m - m0);
                    let col = x.im2col(kh, kw, (sh, sw), (ph, pw), out_h, out_w, m0, mc)?;
                    parts.push(col.matmul(&wt2)?);
                    m0 += mc;
                }
                let refs: Vec<&Tensor> = parts.iter().collect();
                Tensor::cat(&refs, 0)?
            };
            out2d.conv_epilogue(bs.as_ref(), Some(&res), None, b, c_out, out_h, out_w)
        })();
        match via_im2col {
            Ok(out) => return Ok(out),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
        // Fallback (CPU/нет im2col): обычный conv2d + add.
        let conv_out = self.conv2d(weight, bias, stride, padding)?;
        conv_out.add(&res)
    }

    /// Conv2d через backend. На CUDA — im2col + GEMM (быстро); fallback direct-conv
    /// kernel (один launch). `self` = input `[B,C_in,H,W]`, `weight`
    /// `[C_out,C_in,Kh,Kw]`, `bias` опц. `[C_out]`. Только dilation=1. Возвращает
    /// `Unsupported`, если backend не умеет (CPU) — `synaptix-ops::conv2d` тогда
    /// падает в generic-путь.
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        if self.rank() != 4 || weight.rank() != 4 {
            return Err(SynaptixError::Unsupported("conv2d: input/weight rank != 4"));
        }
        let (b, c_in, h, w) = (self.dims()[0], self.dims()[1], self.dims()[2], self.dims()[3]);
        let (c_out, c_in_w, kh, kw) =
            (weight.dims()[0], weight.dims()[1], weight.dims()[2], weight.dims()[3]);
        if c_in != c_in_w {
            return Err(SynaptixError::Unsupported("conv2d: c_in mismatch"));
        }
        if self.dtype() != weight.dtype() {
            return Err(SynaptixError::Unsupported("conv2d: dtype mismatch x/w"));
        }
        let (sh, sw) = (stride.0.max(1), stride.1.max(1));
        let (ph, pw) = padding;
        if h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(SynaptixError::Unsupported("conv2d: input too small"));
        }
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let wt = if weight.is_contiguous() { weight.clone() } else { weight.contiguous()? };
        let bs = match bias {
            Some(b_t) => {
                if b_t.rank() != 1 || b_t.dims()[0] != c_out || b_t.dtype() != self.dtype() {
                    return Err(SynaptixError::Unsupported("conv2d: bias shape/dtype"));
                }
                Some(if b_t.is_contiguous() { b_t.clone() } else { b_t.contiguous()? })
            }
            None => None,
        };

        // Быстрый путь: im2col + GEMM (CUDA с cutlass). При Unsupported (CPU нет
        // im2col / нет cutlass-GEMM) / NonContiguous — падаем в direct-conv ниже.
        match x.conv2d_im2col(&wt, bs.as_ref(), (sh, sw), (ph, pw), out_h, out_w) {
            Ok(out) => return Ok(out),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }

        let out_layout =
            Layout::contiguous(Shape::new(vec![b, c_out, out_h, out_w]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.conv2d(
            (&x.storage, &x.layout),
            (&wt.storage, &wt.layout),
            bs.as_ref().map(|t| (&*t.storage, &t.layout)),
            (&mut storage, &out_layout),
            (sh, sw),
            (ph, pw),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Direct conv3d через backend (dilation=1). `input` `[B,C_in,D,H,W]`,
    /// `weight` `[C_out,C_in,Kd,Kh,Kw]` → `[B,C_out,D_out,H_out,W_out]`. Backend
    /// без поддержки (CPU) → `Unsupported` → caller (`synaptix-ops::conv3d`)
    /// падает в decomposed путь.
    pub fn conv3d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Result<Self> {
        if self.rank() != 5 || weight.rank() != 5 {
            return Err(SynaptixError::Unsupported("conv3d: input/weight rank != 5"));
        }
        let (b, c_in, dz, h, w) =
            (self.dims()[0], self.dims()[1], self.dims()[2], self.dims()[3], self.dims()[4]);
        let (c_out, c_in_w, kd, kh, kw) =
            (weight.dims()[0], weight.dims()[1], weight.dims()[2], weight.dims()[3], weight.dims()[4]);
        if c_in != c_in_w {
            return Err(SynaptixError::Unsupported("conv3d: c_in mismatch"));
        }
        if self.dtype() != weight.dtype() {
            return Err(SynaptixError::Unsupported("conv3d: dtype mismatch x/w"));
        }
        let (sd, sh, sw) = (stride.0.max(1), stride.1.max(1), stride.2.max(1));
        let (pd, ph, pw) = padding;
        if dz + 2 * pd < kd || h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(SynaptixError::Unsupported("conv3d: input too small"));
        }
        let out_d = (dz + 2 * pd - kd) / sd + 1;
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let wt = if weight.is_contiguous() { weight.clone() } else { weight.contiguous()? };
        let bs = match bias {
            Some(b_t) => {
                if b_t.rank() != 1 || b_t.dims()[0] != c_out || b_t.dtype() != self.dtype() {
                    return Err(SynaptixError::Unsupported("conv3d: bias shape/dtype"));
                }
                Some(if b_t.is_contiguous() { b_t.clone() } else { b_t.contiguous()? })
            }
            None => None,
        };
        // Быстрый путь (CUDA, sd==1): разложение по временно́му ядру в `kd`
        // 2D-свёрток (im2col+GEMM, tensor-core/tiled) — даёт ~10-20× над
        // direct-voxel-ядром на больших C (reuse в GEMM vs zero-reuse в naive).
        // out[:,:,do] = Σ_kd conv2d(xp[:,:,kd+do], W[:,:,kd]); xp — temporal
        // zero-pad на pd. Unsupported (CPU нет conv2d) → direct-ядро ниже.
        if sd == 1 && matches!(self.device(), Device::Cuda(_)) {
            match Self::conv3d_via_conv2d(&x, &wt, bs.as_ref(), kd, (sh, sw), (pd, ph, pw), out_d) {
                Ok(out) => return Ok(out),
                Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
                Err(e) => return Err(e),
            }
        }
        let out_layout =
            Layout::contiguous(Shape::new(vec![b, c_out, out_d, out_h, out_w]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.conv3d(
            (&x.storage, &x.layout),
            (&wt.storage, &wt.layout),
            bs.as_ref().map(|t| (&*t.storage, &t.layout)),
            (&mut storage, &out_layout),
            (sd, sh, sw),
            (pd, ph, pw),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Depthwise conv1d (groups == C): `self [B,C,L]`, `weight [C,1,K]`,
    /// `bias [C]?`. `transpose=false` → stride-свёртка с zero-pad `padding`,
    /// out `[(L+2p−K)/s+1]`; `true` → conv_transpose полной длины `(L−1)·s+K`
    /// (кроп по padding у вызывающего). Один thread = один выходной элемент.
    pub fn dwconv1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize,
        transpose: bool,
    ) -> Result<Self> {
        if self.rank() != 3 || weight.rank() != 3 || weight.dims()[1] != 1 {
            return Err(SynaptixError::Unsupported("dwconv1d: input [B,C,L], weight [C,1,K]"));
        }
        let (b, c, l) = (self.dims()[0], self.dims()[1], self.dims()[2]);
        let k = weight.dims()[2];
        if weight.dims()[0] != c || self.dtype() != weight.dtype() {
            return Err(SynaptixError::Unsupported("dwconv1d: weight C/dtype mismatch"));
        }
        let s = stride.max(1);
        let lo = if transpose {
            (l - 1) * s + k
        } else {
            if l + 2 * padding < k {
                return Err(SynaptixError::Unsupported("dwconv1d: input too small"));
            }
            (l + 2 * padding - k) / s + 1
        };
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let wt = if weight.is_contiguous() { weight.clone() } else { weight.contiguous()? };
        let bs = match bias {
            Some(b_t) => {
                if b_t.rank() != 1 || b_t.dims()[0] != c || b_t.dtype() != self.dtype() {
                    return Err(SynaptixError::Unsupported("dwconv1d: bias shape/dtype"));
                }
                Some(if b_t.is_contiguous() { b_t.clone() } else { b_t.contiguous()? })
            }
            None => None,
        };
        let out_layout = Layout::contiguous(Shape::new(vec![b, c, lo]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.dwconv1d(
            (&x.storage, &x.layout),
            (&wt.storage, &wt.layout),
            bs.as_ref().map(|t| (&*t.storage, &t.layout)),
            (&mut storage, &out_layout),
            s,
            padding,
            transpose,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// NHWC-насквозь conv3d (B=1, CUDA, F16/BF16, C%8): вход транспонируется в
    /// NHWC ОДИН раз, kd-слайсы — offset-view без копий, Σ_kd аккумулируется
    /// fused residual-эпилогом implicit-GEMM (bias на kd=0), kd-фильтры — из
    /// KRSC-кэша по исходному 5D-весу. Старый путь платил на КАЖДЫЙ kd narrow+
    /// permute+contiguous входа, nchw_to_nhwc, отдельное add-ядро и
    /// переконвертацию веса (krsc-кэш мазал по временному указателю): ~28 байт
    /// DRAM-трафика на байт GEMM-полезных против ~12 здесь.
    #[allow(clippy::too_many_arguments)]
    fn conv3d_nhwc_fast(
        xp: &Tensor,
        wt: &Tensor,
        bias: Option<&Tensor>,
        kd_size: usize,
        stride_hw: (usize, usize),
        pad_hw: (usize, usize),
        out_d: usize,
    ) -> Result<Tensor> {
        let (c_in, dp, h, w) =
            (xp.dims()[1], xp.dims()[2], xp.dims()[3], xp.dims()[4]);
        let (c_out, kh, kw) = (wt.dims()[0], wt.dims()[3], wt.dims()[4]);
        let (sh, sw) = stride_hw;
        let (ph, pw) = pad_hw;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;
        if let Some(bt) = bias {
            if !bt.is_contiguous() || bt.dtype() != xp.dtype() || bt.layout.numel() != c_out {
                return Err(SynaptixError::Unsupported("conv3d_nhwc: bias"));
            }
        }
        // [1,C,Dp,H,W] → [1,C,Dp,H·W] → NHWC [1,Dp,H·W,C] → [Dp,H,W,C]
        let xp_nhwc = xp
            .reshape(vec![1, c_in, dp, h * w])?
            .nchw_to_nhwc()?
            .reshape(vec![dp, h, w, c_in])?;
        let wkd = cached_conv3d_wkd_krsc(wt, kd_size)?;
        let backend = registry::backend_for(xp.device())?;
        let stream = Stream::default_for(xp.device())?;
        const CHUNK: usize = 16;
        let mut chunks: Vec<Tensor> = Vec::with_capacity(out_d.div_ceil(CHUNK));
        let mut d0 = 0usize;
        while d0 < out_d {
            let cd = CHUNK.min(out_d - d0);
            let mut acc: Option<Tensor> = None;
            for kdi in 0..kd_size {
                let x_b = xp_nhwc.narrow(0, kdi + d0, cd)?; // offset-view, strides contiguous
                let out_layout =
                    Layout::contiguous(Shape::new(vec![cd, oh, ow, c_out]), xp.dtype());
                let out_bytes = xp.dtype().bytes_for_numel(out_layout.numel());
                let mut storage = backend.alloc_uninit(out_bytes, xp.device())?;
                let bias_arg = if kdi == 0 {
                    bias.map(|t| (&*t.storage, &t.layout))
                } else {
                    None
                };
                let res_arg = acc.as_ref().map(|t: &Tensor| (&*t.storage, &t.layout));
                backend.conv2d_implicit_nhwc(
                    (&x_b.storage, &x_b.layout),
                    (&wkd[kdi].storage, &wkd[kdi].layout),
                    bias_arg,
                    res_arg,
                    None,
                    (&mut storage, &out_layout),
                    true,
                    stride_hw,
                    pad_hw,
                    &stream,
                )?;
                acc = Some(Tensor::from_parts(Arc::new(storage), out_layout));
            }
            chunks.push(acc.ok_or(SynaptixError::Unsupported("conv3d_nhwc: Kd=0"))?);
            d0 += cd;
        }
        let refs: Vec<&Tensor> = chunks.iter().collect();
        let nhwc = if refs.len() == 1 { chunks[0].clone() } else { Tensor::cat(&refs, 0)? };
        // [D,oh,ow,C] → [1,D,oh·ow,C] → NCHW [1,C,D,oh·ow] → [1,C,D,oh,ow]
        nhwc.reshape(vec![1, out_d, oh * ow, c_out])?
            .nhwc_to_nchw()?
            .reshape(vec![1, c_out, out_d, oh, ow])
    }

    /// conv3d через `kd` батч-2D-свёрток (см. [`Tensor::conv3d`]). `x`
    /// `[B,Cin,D,H,W]`, `wt` `[Cout,Cin,Kd,Kh,Kw]` (оба contiguous), sd=dd=1.
    /// `out[:,:,do] = Σ_kd conv2d(xp[:,:,kd+do], wt[:,:,kd])`, xp = temporal
    /// zero-pad `x` на `pd`. bias добавляется один раз в конце.
    #[allow(clippy::too_many_arguments)]
    fn conv3d_via_conv2d(
        x: &Tensor,
        wt: &Tensor,
        bias: Option<&Tensor>,
        kd_size: usize,
        stride_hw: (usize, usize),
        padding: (usize, usize, usize),
        out_d: usize,
    ) -> Result<Tensor> {
        let (b, c_in, _d, h, w) =
            (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3], x.dims()[4]);
        let c_out = wt.dims()[0];
        let (pd, ph, pw) = padding;
        let xp = if pd > 0 {
            let z = Tensor::zeros(vec![b, c_in, pd, h, w], x.dtype(), x.device())?;
            Tensor::cat(&[&z, x, &z], 2)?
        } else {
            x.clone()
        };
        if b == 1
            && xp.device().is_cuda()
            && matches!(xp.dtype(), DType::F16 | DType::BF16)
            && c_in % 8 == 0
            && c_out % 8 == 0
        {
            match Self::conv3d_nhwc_fast(
                &xp, wt, bias, kd_size, stride_hw, (ph, pw), out_d,
            ) {
                Ok(t) => return Ok(t),
                Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
                Err(e) => return Err(e),
            }
        }
        // Тайлинг по out_d (чанки кадров): на FullHD батч всех кадров разом
        // (B*out_d) даёт крупные сосуществующие тензоры → OOM. Чанк ограничивает
        // пик памяти конволюции, сохраняя GEMM-скорость; швов нет — каждый
        // выходной кадр считается точно через kd-цикл. На малом out_d (≤CHUNK) —
        // одна итерация (поведение без тайлинга).
        const CHUNK: usize = 16;
        let mut chunks: Vec<Tensor> = Vec::with_capacity(out_d.div_ceil(CHUNK));
        let mut d0 = 0usize;
        while d0 < out_d {
            let cd = CHUNK.min(out_d - d0);
            // выходные кадры [d0, d0+cd): Σ_kd conv2d(xp[:,:,kdi+d0:+cd], W[:,:,kdi])
            let mut acc: Option<Tensor> = None;
            let (mut oh, mut ow) = (0usize, 0usize);
            for kdi in 0..kd_size {
                let x_b = xp
                    .narrow(2, kdi + d0, cd)?
                    .permute(vec![0, 2, 1, 3, 4])?
                    .contiguous()?
                    .reshape(vec![b * cd, c_in, h, w])?;
                let w_kd = wt.narrow(2, kdi, 1)?.squeeze(2)?.contiguous()?; // [Cout,Cin,Kh,Kw]
                let y = x_b.conv2d(&w_kd, None, stride_hw, (ph, pw))?; // [B*cd,Cout,oh,ow]
                oh = y.dims()[2];
                ow = y.dims()[3];
                acc = Some(match acc.take() {
                    Some(a) => a.add(&y)?,
                    None => y,
                });
            }
            let acc = acc.ok_or(SynaptixError::Unsupported("conv3d: Kd=0"))?;
            chunks.push(
                acc.reshape(vec![b, cd, c_out, oh, ow])?
                    .permute(vec![0, 2, 1, 3, 4])?
                    .contiguous()?,
            ); // [B,Cout,cd,oh,ow]
            d0 += cd;
        }
        let refs: Vec<&Tensor> = chunks.iter().collect();
        let mut out = if refs.len() == 1 { refs[0].clone() } else { Tensor::cat(&refs, 2)? };
        if let Some(bt) = bias {
            out = out.broadcast_add(&bt.reshape(vec![1, c_out, 1, 1, 1])?)?;
        }
        Ok(out)
    }

    /// Fused PixelNorm (+опц. silu): `out = x / sqrt(mean(x², dim=1)+eps)`
    /// per-location по канальной оси NCHW `[B,C,*spatial]`. F32-аккумулятор в
    /// ядре. `Unsupported` (CPU/dtype) → caller падает в decomposed путь.
    pub fn pixel_norm_fused(&self, eps: f32, silu: bool) -> Result<Self> {
        if self.rank() < 2 {
            return Err(SynaptixError::Unsupported("pixel_norm_fused: rank < 2"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(self.dims().to_vec()), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.pixel_norm(
            (&x.storage, &x.layout),
            (&mut storage, &out_layout),
            self.dims()[1],
            eps,
            silu,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fused LayerNorm через backend: `out = ((self-mean)/sqrt(var+eps))*weight [+bias]`
    /// по last dim. `weight`/`bias` — `[H]`. Возвращает `Unsupported`, если backend
    /// не умеет (CPU) — `synaptix-ops::layer_norm` тогда падает в decomposed путь.
    pub fn layer_norm_fused(&self, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Self> {
        let xr = self.rank();
        if xr == 0 {
            return Err(SynaptixError::Unsupported("layer_norm_fused: scalar x"));
        }
        let h = self.dims()[xr - 1];
        if weight.rank() != 1 || weight.dims()[0] != h {
            return Err(SynaptixError::Unsupported("layer_norm_fused: weight shape"));
        }
        if self.dtype() != weight.dtype() {
            return Err(SynaptixError::Unsupported("layer_norm_fused: dtype mismatch"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let w = if weight.is_contiguous() { weight.clone() } else { weight.contiguous()? };
        let b = match bias {
            Some(b) => {
                if b.rank() != 1 || b.dims()[0] != h || b.dtype() != self.dtype() {
                    return Err(SynaptixError::Unsupported("layer_norm_fused: bias shape/dtype"));
                }
                Some(if b.is_contiguous() { b.clone() } else { b.contiguous()? })
            }
            None => None,
        };
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.layer_norm(
            (&x.storage, &x.layout),
            (&w.storage, &w.layout),
            b.as_ref().map(|t| (&*t.storage, &t.layout)),
            (&mut storage, &out_layout),
            eps,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fused Split RoPE через backend. `self`: `[.., S, D]`; `cos`/`sin`: F32
    /// `[S, D/2]` (как из `RopeCache::select_*`). Возвращает `Unsupported`, если
    /// backend не умеет — `synaptix-ops::apply_rope` тогда падает в decomposed.
    pub fn rope_split_fused(&self, cos: &Tensor, sin: &Tensor) -> Result<Self> {
        let xr = self.rank();
        if xr < 2 {
            return Err(SynaptixError::Unsupported("rope_split_fused: x rank < 2"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let cos_c = if cos.is_contiguous() { cos.clone() } else { cos.contiguous()? };
        let sin_c = if sin.is_contiguous() { sin.clone() } else { sin.contiguous()? };
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.rope_split(
            (&x.storage, &x.layout),
            (&cos_c.storage, &cos_c.layout),
            (&sin_c.storage, &sin_c.layout),
            (&mut storage, &out_layout),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Partial Split RoPE одним fused-ядром: вращает первые `rot_dim` из `D`,
    /// остальные измерения проходят без изменений (MiniMax-H3 MM-RoPE: 96 из 128).
    /// `self` [.., S, D]; `cos`/`sin` — F32 `[S, rot_dim/2]`. Позиция строки =
    /// `row % S`, поэтому при layout `[H, S, D]` таблица broadcast'ится по головам
    /// без материализации `[H*S, ..]`. `Unsupported` если backend не умеет.
    pub fn rope_split_partial_fused(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        rot_dim: usize,
    ) -> Result<Self> {
        let xr = self.rank();
        if xr < 2 {
            return Err(SynaptixError::Unsupported("rope_split_partial_fused: x rank < 2"));
        }
        let d = self.dims()[xr - 1];
        if rot_dim == 0 || rot_dim % 2 != 0 || rot_dim > d {
            return Err(SynaptixError::Unsupported("rope_split_partial_fused: rot_dim"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let cos_c = if cos.is_contiguous() { cos.clone() } else { cos.contiguous()? };
        let sin_c = if sin.is_contiguous() { sin.clone() } else { sin.contiguous()? };
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.rope_split_partial(
            (&x.storage, &x.layout),
            (&cos_c.storage, &cos_c.layout),
            (&sin_c.storage, &sin_c.layout),
            rot_dim,
            (&mut storage, &out_layout),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Interleaved (adjacent-pair / FLUX) RoPE одним fused-ядром. `self` [B,S,H,D];
    /// `cos`/`sin` — F32 ПОЛНАЯ таблица с numel=S*D (любой shape). Заменяет ~10
    /// decomposed-ops apply_rope. `Unsupported` если backend не умеет → fallback.
    pub fn rope_interleaved_fused(&self, cos: &Tensor, sin: &Tensor) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("rope_interleaved_fused: x rank != 4"));
        }
        let h = self.dims()[2];
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let cos_c = if cos.is_contiguous() { cos.clone() } else { cos.contiguous()? };
        let sin_c = if sin.is_contiguous() { sin.clone() } else { sin.contiguous()? };
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.rope_interleaved(
            (&x.storage, &x.layout),
            (&cos_c.storage, &cos_c.layout),
            (&sin_c.storage, &sin_c.layout),
            h,
            (&mut storage, &out_layout),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Fused attention: `out = softmax(scale·selfᵀ·Kᵀ [+causal])·V`. `self` (Q)
    /// `[B,NH,Tq,D]`, `k`/`v` `[B,NKV,Tkv,D]` (GQA через backend, без repeat_kv).
    /// Возвращает `Unsupported`, если backend не умеет — вызывающий код тогда
    /// падает в repeat_kv + scaled_dot_attention.
    pub fn flash_attention_bshd(
        &self,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("flash_bshd: q rank != 4"));
        }
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let k_c = if k.is_contiguous() { k.clone() } else { k.contiguous()? };
        let v_c = if v.is_contiguous() { v.clone() } else { v.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(q.dims().to_vec()), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention_bshd(
            (&q.storage, &q.layout),
            (&k_c.storage, &k_c.layout),
            (&v_c.storage, &v_c.layout),
            (&mut storage, &out_layout),
            scale,
            causal,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Top-k по строкам `[rows, cols]`: возвращает `(значения, индексы)` формы
    /// `[rows, k]`, отсортированные по убыванию.
    pub fn topk_rows(&self, k: usize) -> Result<(Self, Self)> {
        if self.rank() != 2 {
            return Err(SynaptixError::Unsupported("topk_rows: ждём [rows, cols]"));
        }
        let (rows, cols) = (self.dims()[0], self.dims()[1]);
        if k == 0 || k > cols {
            return Err(SynaptixError::Unsupported("topk_rows: k вне ширины строки"));
        }
        let src = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let idx_layout = Layout::contiguous(Shape::new(vec![rows, k]), DType::U32);
        let val_layout = Layout::contiguous(Shape::new(vec![rows, k]), self.dtype());
        let backend = registry::backend_for(self.device())?;
        let mut idx = backend.alloc_uninit(DType::U32.bytes_for_numel(rows * k), self.device())?;
        let mut val = backend.alloc_uninit(self.dtype().bytes_for_numel(rows * k), self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.topk_rows(
            (&src.storage, &src.layout),
            (&mut idx, &idx_layout),
            (&mut val, &val_layout),
            k,
            &stream,
        )?;
        Ok((
            Tensor::from_parts(Arc::new(val), val_layout),
            Tensor::from_parts(Arc::new(idx), idx_layout),
        ))
    }

    /// Top-k по широким строкам: `valid` — сколько первых столбцов строки
    /// действительны. Возвращает индексы `[rows, k]`, где незанятые слоты
    /// помечены `u32::MAX`.
    pub fn topk_wide(&self, valid: &Tensor, k: usize) -> Result<Self> {
        if self.rank() != 2 {
            return Err(SynaptixError::Unsupported("topk_wide: ждём [rows, cols]"));
        }
        let (rows, cols) = (self.dims()[0], self.dims()[1]);
        if k == 0 || k > cols {
            return Err(SynaptixError::Unsupported("topk_wide: k вне ширины строки"));
        }
        let src = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![rows, k]), DType::U32);
        let backend = registry::backend_for(self.device())?;
        let mut out = backend.alloc_uninit(DType::U32.bytes_for_numel(rows * k), self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.topk_wide(
            (&src.storage, &src.layout),
            (&valid.storage, &valid.layout),
            (&mut out, &out_layout),
            k,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(out), out_layout))
    }

    /// Attention по таблице блоков KV: `self` — запросы `[B, NH, D]`, `k`/`v` —
    /// общий буфер слоя `[NKV, CAP, D]`, `table` — индексы блоков `[B, NB]`,
    /// `tail_from`/`tail_len` — хвост каждого запроса `[rows]`. Позиция блока
    /// `b` это `b · ratio`, а `row_offset` говорит, с какой строки таблицы
    /// начинается этот запрос: срез по строкам пришлось бы материализовать.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_blocks(
        &self,
        k: &Tensor,
        v: &Tensor,
        table: &Tensor,
        tail_from: &Tensor,
        tail_len: &Tensor,
        ratio: usize,
        scale: f32,
        row_offset: usize,
    ) -> Result<Self> {
        if self.rank() != 3 {
            return Err(SynaptixError::Unsupported("flash_blocks: q rank != 3"));
        }
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let k_c = if k.is_contiguous() { k.clone() } else { k.contiguous()? };
        let v_c = if v.is_contiguous() { v.clone() } else { v.contiguous()? };
        let tb = if table.is_contiguous() { table.clone() } else { table.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(q.dims().to_vec()), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention_blocks(
            (&q.storage, &q.layout),
            (&k_c.storage, &k_c.layout),
            (&v_c.storage, &v_c.layout),
            (&tb.storage, &tb.layout),
            (&tail_from.storage, &tail_from.layout),
            (&tail_len.storage, &tail_len.layout),
            (&mut storage, &out_layout),
            ratio,
            scale,
            row_offset,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_blocks_mxfp8(
        &self,
        k: &Tensor,
        v: &Tensor,
        k_scale: &Tensor,
        v_scale: &Tensor,
        table: &Tensor,
        tail_from: &Tensor,
        tail_len: &Tensor,
        ratio: usize,
        scale: f32,
        row_offset: usize,
    ) -> Result<Self> {
        if self.rank() != 3 {
            return Err(SynaptixError::Unsupported("flash_blocks mxfp8: q rank != 3"));
        }
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let tb = if table.is_contiguous() { table.clone() } else { table.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(q.dims().to_vec()), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention_blocks_mxfp8(
            (&q.storage, &q.layout),
            (&k.storage, &k.layout),
            (&v.storage, &v.layout),
            (&k_scale.storage, &k_scale.layout),
            (&v_scale.storage, &v_scale.layout),
            (&tb.storage, &tb.layout),
            (&tail_from.storage, &tail_from.layout),
            (&tail_len.storage, &tail_len.layout),
            (&mut storage, &out_layout),
            ratio,
            scale,
            row_offset,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    pub fn mxfp8_dequant(&self, scales: &Tensor) -> Result<Self> {
        if self.dtype() != DType::MXFP8 {
            return Err(SynaptixError::Unsupported("mxfp8_dequant: тензор не MXFP8"));
        }
        let packed = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let sc = if scales.is_contiguous() { scales.clone() } else { scales.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(self.dims().to_vec()), DType::F16);
        let out_bytes = DType::F16.bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_uninit(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.mxfp8_dequant(
            (&packed.storage, &packed.layout),
            (&sc.storage, &sc.layout),
            (&mut storage, &out_layout),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    pub fn flash_attention(
        &self,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("flash_attention: q rank != 4"));
        }
        let dims = self.dims();
        let (b, nh, t_q, d) = (dims[0], dims[1], dims[2], dims[3]);
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        // K/V могут быть strided view preallocated KV-буфера `[B,nkv,max_seq,hd]`,
        // суженного по dim-T до активной длины (narrow(2,0,seq_len)). Backend выводит
        // physical t_stride из layout — НЕ материализуем contiguous (это была бы O(S)
        // копия всего KV каждый decode-токен). Иначе (неподдержанный layout) — contiguous.
        let k_c = if kv_layout_passthrough_ok(&k.layout) { k.clone() } else { k.contiguous()? };
        let v_c = if kv_layout_passthrough_ok(&v.layout) { v.clone() } else { v.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![b, nh, t_q, d]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention(
            (&q.storage, &q.layout),
            (&k_c.storage, &k_c.layout),
            (&v_c.storage, &v_c.layout),
            (&mut storage, &out_layout),
            scale,
            causal,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    pub fn flash_attention_window(
        &self,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        window: i32,
        causal: bool,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("flash_attention_window: q rank != 4"));
        }
        let dims = self.dims();
        let (b, nh, t_q, d) = (dims[0], dims[1], dims[2], dims[3]);
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let k_c = if kv_layout_passthrough_offset_ok(&k.layout) { k.clone() } else { k.contiguous()? };
        let v_c = if kv_layout_passthrough_offset_ok(&v.layout) { v.clone() } else { v.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![b, nh, t_q, d]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention_window(
            (&q.storage, &q.layout),
            (&k_c.storage, &k_c.layout),
            (&v_c.storage, &v_c.layout),
            (&mut storage, &out_layout),
            scale,
            window,
            causal,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// In-place append `src` `[B, nkv, T_new, hd]` в preallocated self-буфер
    /// `[B, nkv, max_seq, hd]` на позицию `seq_pos` по dim-T. Без аллокации/cat —
    /// O(T_new) запись вместо O(S) реаллокации всего KV-буфера. Требует
    /// уникального владения storage (`self` не должен быть aliased view —
    /// иначе `Arc::get_mut` вернёт None).
    pub fn kv_append_inplace(&mut self, src: &Tensor, seq_pos: usize) -> Result<()> {
        if self.device() != src.device() {
            return Err(SynaptixError::device_mismatch(self.device(), src.device()));
        }
        let src_c = if src.is_contiguous() { src.clone() } else { src.contiguous()? };
        let backend = registry::backend_for(self.device())?;
        let stream = Stream::default_for(self.device())?;
        let layout = self.layout.clone();
        let storage = Arc::get_mut(&mut self.storage).ok_or_else(|| {
            SynaptixError::Other(
                "kv_append_inplace: storage aliased (KV-буфер должен быть уникально владеемым)"
                    .into(),
            )
        })?;
        backend.kv_append((storage, &layout), (&src_c.storage, &src_c.layout), seq_pos, &stream)
    }

    /// MXFP8-KV flash-attention (block-scale). `self` — float Q `[B,NH,Tq,D]`;
    /// `k`/`v` — MXFP8 `[B,NKV,Tkv,D]`; `k_scale`/`v_scale` — U8 E8M0
    /// `[B,NKV,Tkv,D/32]`. Деквант per-32-block inline.
    pub fn flash_attention_mxfp8kv(
        &self,
        k: &Tensor,
        v: &Tensor,
        k_scale: &Tensor,
        v_scale: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("flash_attention_mxfp8kv: q rank != 4"));
        }
        let dims = self.dims();
        let (b, nh, t_q, d) = (dims[0], dims[1], dims[2], dims[3]);
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![b, nh, t_q, d]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention_mxfp8kv(
            (&q.storage, &q.layout),
            (&k.storage, &k.layout),
            (&v.storage, &v.layout),
            (&k_scale.storage, &k_scale.layout),
            (&v_scale.storage, &v_scale.layout),
            (&mut storage, &out_layout),
            scale,
            causal,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// MXFP8-KV квантизующий in-place append: `self` — MXFP8 буфер
    /// `[B,nkv,max_seq,hd]`, `scale` — U8 E8M0 буфер `[B,nkv,max_seq,hd/32]`,
    /// `src` — BF16 `[B,nkv,T_new,hd]`. Квантизует `src` (per-32-block amax→E8M0)
    /// в slot `seq_pos`. Оба буфера должны быть уникально владеемы.
    pub fn kv_append_quant_mxfp8_inplace(
        &mut self,
        scale: &mut Tensor,
        src: &Tensor,
        seq_pos: usize,
    ) -> Result<()> {
        if self.device() != src.device() || self.device() != scale.device() {
            return Err(SynaptixError::device_mismatch(self.device(), src.device()));
        }
        let src_c = if src.is_contiguous() { src.clone() } else { src.contiguous()? };
        let backend = registry::backend_for(self.device())?;
        let stream = Stream::default_for(self.device())?;
        let dst_layout = self.layout.clone();
        let scale_layout = scale.layout.clone();
        let dst_storage = Arc::get_mut(&mut self.storage).ok_or_else(|| {
            SynaptixError::Other("kv_append_quant_mxfp8_inplace: MXFP8-буфер aliased".into())
        })?;
        let scale_storage = Arc::get_mut(&mut scale.storage).ok_or_else(|| {
            SynaptixError::Other("kv_append_quant_mxfp8_inplace: scale-буфер aliased".into())
        })?;
        backend.kv_append_quant_mxfp8(
            (dst_storage, &dst_layout),
            (scale_storage, &scale_layout),
            (&src_c.storage, &src_c.layout),
            seq_pos,
            &stream,
        )
    }

    /// Device-resident-position Split RoPE (CUDA-graph decode). `self`
    /// `[b,h,t,head_dim]`; `cos`/`sin` — dtype `self`, дублированный layout
    /// `[max_seq, rotary_dim]` (см. [`Backend::rope_apply_dev`]); `start_pos` —
    /// U32[1] device-резидент. Возвращает повёрнутый тензор (новая аллокация).
    pub fn rope_apply_dev(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        start_pos: &Tensor,
        rotary_dim: usize,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("rope_apply_dev: x rank != 4"));
        }
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let cos_c = if cos.is_contiguous() { cos.clone() } else { cos.contiguous()? };
        let sin_c = if sin.is_contiguous() { sin.clone() } else { sin.contiguous()? };
        let out_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.rope_apply_dev(
            (&x.storage, &x.layout),
            (&cos_c.storage, &cos_c.layout),
            (&sin_c.storage, &sin_c.layout),
            (&start_pos.storage, &start_pos.layout),
            (&mut storage, &out_layout),
            rotary_dim,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Device-resident-position in-place KV append (CUDA-graph decode). Как
    /// [`Tensor::kv_append_inplace`], но слот `seq_pos` — U32[1] device-резидент.
    pub fn kv_append_dev(&mut self, src: &Tensor, seq_pos: &Tensor) -> Result<()> {
        if self.device() != src.device() {
            return Err(SynaptixError::device_mismatch(self.device(), src.device()));
        }
        let src_c = if src.is_contiguous() { src.clone() } else { src.contiguous()? };
        let backend = registry::backend_for(self.device())?;
        let stream = Stream::default_for(self.device())?;
        let layout = self.layout.clone();
        let sp_storage = &seq_pos.storage;
        let sp_layout = &seq_pos.layout;
        let storage = Arc::get_mut(&mut self.storage).ok_or_else(|| {
            SynaptixError::Other("kv_append_dev: storage aliased (KV-буфер должен быть уникально владеемым)".into())
        })?;
        backend.kv_append_dev(
            (storage, &layout),
            (&src_c.storage, &src_c.layout),
            (sp_storage, sp_layout),
            &stream,
        )
    }

    /// Device-resident-length FA-prefill (CUDA-graph prefill chunk'а).
    /// `self` (Q) `[B,NH,Tq,D]`, Tq>1; `k`/`v` — полный preallocated буфер
    /// `[B,nkv,max_seq,hd]` (t_stride выводится из layout); `t_cache` — U32[1]
    /// device-резидент (активная длина KV после append'а chunk'а, т.е.
    /// `pos_start + Tq`). Tensor-core FA-4 ядро с Q-тайлингом (BM=16, WMMA);
    /// для Tq=256 ~4× быстрее `flash_attention_dev` (тот = decode split-K без
    /// Q-тайлинга, годится только для Tq=1).
    pub fn flash_attention_prefill_dev(
        &self,
        k: &Tensor,
        v: &Tensor,
        t_cache: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("flash_attention_prefill_dev: q rank != 4"));
        }
        let dims = self.dims();
        let (b, nh, t_q, d) = (dims[0], dims[1], dims[2], dims[3]);
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let k_c = if kv_layout_passthrough_ok(&k.layout) { k.clone() } else { k.contiguous()? };
        let v_c = if kv_layout_passthrough_ok(&v.layout) { v.clone() } else { v.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![b, nh, t_q, d]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention_prefill_dev(
            (&q.storage, &q.layout),
            (&k_c.storage, &k_c.layout),
            (&v_c.storage, &v_c.layout),
            (&t_cache.storage, &t_cache.layout),
            (&mut storage, &out_layout),
            scale,
            causal,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Device-resident-length flash-decode (CUDA-graph decode). `self` (Q)
    /// `[B,NH,Tq,D]`; `k`/`v` — полный preallocated буфер `[B,nkv,max_seq,hd]`
    /// (t_stride выводится из layout); `t_cache` — U32[1] device-резидент
    /// (активная длина KV). Возвращает `[B,NH,Tq,D]`.
    pub fn flash_attention_dev(
        &self,
        k: &Tensor,
        v: &Tensor,
        t_cache: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("flash_attention_dev: q rank != 4"));
        }
        let dims = self.dims();
        let (b, nh, t_q, d) = (dims[0], dims[1], dims[2], dims[3]);
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let k_c = if kv_layout_passthrough_ok(&k.layout) { k.clone() } else { k.contiguous()? };
        let v_c = if kv_layout_passthrough_ok(&v.layout) { v.clone() } else { v.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![b, nh, t_q, d]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention_dev(
            (&q.storage, &q.layout),
            (&k_c.storage, &k_c.layout),
            (&v_c.storage, &v_c.layout),
            (&t_cache.storage, &t_cache.layout),
            (&mut storage, &out_layout),
            scale,
            causal,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    pub fn flash_attention_window_dev(
        &self,
        k: &Tensor,
        v: &Tensor,
        t_cache: &Tensor,
        scale: f32,
        window: i32,
        causal: bool,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("flash_attention_window_dev: q rank != 4"));
        }
        let dims = self.dims();
        let (b, nh, t_q, d) = (dims[0], dims[1], dims[2], dims[3]);
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let k_c = if kv_layout_passthrough_ok(&k.layout) { k.clone() } else { k.contiguous()? };
        let v_c = if kv_layout_passthrough_ok(&v.layout) { v.clone() } else { v.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![b, nh, t_q, d]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention_window_dev(
            (&q.storage, &q.layout),
            (&k_c.storage, &k_c.layout),
            (&v_c.storage, &v_c.layout),
            (&t_cache.storage, &t_cache.layout),
            (&mut storage, &out_layout),
            scale,
            window,
            causal,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Device-pos MXFP8-KV квантизующий append (CUDA-graph). Как
    /// [`Self::kv_append_quant_mxfp8_inplace`], но `seq_pos` device-резидентный
    /// U32[1] тензор (один граф валиден для всех позиций).
    pub fn kv_append_quant_mxfp8_dev(
        &mut self,
        scale: &mut Tensor,
        src: &Tensor,
        seq_pos: &Tensor,
    ) -> Result<()> {
        if self.device() != src.device() || self.device() != scale.device() {
            return Err(SynaptixError::device_mismatch(self.device(), src.device()));
        }
        let src_c = if src.is_contiguous() { src.clone() } else { src.contiguous()? };
        let backend = registry::backend_for(self.device())?;
        let stream = Stream::default_for(self.device())?;
        let dst_layout = self.layout.clone();
        let scale_layout = scale.layout.clone();
        let sp_storage = &seq_pos.storage;
        let sp_layout = &seq_pos.layout;
        let dst_storage = Arc::get_mut(&mut self.storage).ok_or_else(|| {
            SynaptixError::Other("kv_append_quant_mxfp8_dev: MXFP8-буфер aliased".into())
        })?;
        let scale_storage = Arc::get_mut(&mut scale.storage).ok_or_else(|| {
            SynaptixError::Other("kv_append_quant_mxfp8_dev: scale-буфер aliased".into())
        })?;
        backend.kv_append_quant_mxfp8_dev(
            (dst_storage, &dst_layout),
            (scale_storage, &scale_layout),
            (&src_c.storage, &src_c.layout),
            (sp_storage, sp_layout),
            &stream,
        )
    }

    /// Device-Tkv MXFP8-KV flash-decode (CUDA-graph). Как
    /// [`Self::flash_attention_mxfp8kv`], но активная длина KV `t_cache` U32[1]
    /// device-резидентна. `k`/`v` — полный MXFP8-буфер `[B,nkv,max_seq,hd]`;
    /// `k_scale`/`v_scale` — U8 `[B,nkv,max_seq,hd/32]`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_mxfp8kv_dev(
        &self,
        k: &Tensor,
        v: &Tensor,
        k_scale: &Tensor,
        v_scale: &Tensor,
        t_cache: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(SynaptixError::Unsupported("flash_attention_mxfp8kv_dev: q rank != 4"));
        }
        let dims = self.dims();
        let (b, nh, t_q, d) = (dims[0], dims[1], dims[2], dims[3]);
        let q = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![b, nh, t_q, d]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.flash_attention_mxfp8kv_dev(
            (&q.storage, &q.layout),
            (&k.storage, &k.layout),
            (&v.storage, &v.layout),
            (&k_scale.storage, &k_scale.layout),
            (&v_scale.storage, &v_scale.layout),
            (&t_cache.storage, &t_cache.layout),
            (&mut storage, &out_layout),
            scale,
            causal,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Token-embedding gather: `self` (table `[vocab,dim]`) → `out[t,:] =
    /// self[ids[t],:]`, форма `[n, dim]`. `ids` — U32 `[n]`. На CUDA читает
    /// индексы с device (capture-safe, без `clone_dtoh`); на не-CUDA / при
    /// `Unsupported` падает в `index_select`-путь (`token_embedding`).
    pub fn embed_gather(&self, ids: &Tensor) -> Result<Self> {
        if self.rank() != 2 {
            return Err(SynaptixError::Unsupported("embed_gather: table rank != 2"));
        }
        let dim = self.dims()[1];
        let n = ids.numel();
        let table = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let ids_c = if ids.is_contiguous() { ids.clone() } else { ids.contiguous()? };
        let out_layout = Layout::contiguous(Shape::new(vec![n, dim]), self.dtype());
        let out_bytes = self.dtype().bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(self.device())?;
        let mut storage = backend.alloc_zeros(out_bytes, self.device())?;
        let stream = Stream::default_for(self.device())?;
        backend.embed_gather(
            (&table.storage, &table.layout),
            (&ids_c.storage, &ids_c.layout),
            (&mut storage, &out_layout),
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    /// Неинициализированный contiguous-тензор (alloc_uninit бэкенда) — для
    /// чанк-записи через [`Tensor::copy_rows_from`] без `cat` (cat требует
    /// цельный кусок + копию всех частей; на фрагментированном пуле длинных T
    /// это главный источник OOM).
    pub fn empty_uninit(dims: Vec<usize>, dtype: DType, device: Device) -> Result<Self> {
        let lo = Layout::contiguous(Shape::new(dims), dtype);
        let backend = registry::backend_for(device)?;
        let st = backend.alloc_uninit(dtype.bytes_for_numel(lo.numel()), device)?;
        Ok(Tensor::from_parts(Arc::new(st), lo))
    }

    /// Запись contiguous `src` в строки `[off, off+src.dims()[1])` оси 1
    /// contiguous-тензора `self` ([B≤1-слитая, T, ...] — строки оси 1 линейны
    /// в памяти → сырой D2D). Чанк-сборка выхода без cat.
    pub fn copy_rows_from(&mut self, off: usize, src: &Tensor) -> Result<()> {
        if !self.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if self.dtype() != src.dtype() || self.rank() != src.rank() {
            return Err(SynaptixError::Unsupported("copy_rows_from: dtype/rank"));
        }
        if self.dims()[0] != 1 || src.dims()[0] != 1 {
            return Err(SynaptixError::Unsupported("copy_rows_from: dim0 != 1"));
        }
        let row: usize = self.dims()[2..].iter().product();
        let src_row: usize = src.dims()[2..].iter().product();
        if row != src_row || off + src.dims()[1] > self.dims()[1] {
            return Err(SynaptixError::Unsupported("copy_rows_from: формы/границы"));
        }
        let src_c = if src.is_contiguous() { src.clone() } else { src.contiguous()? };
        let esz = self.dtype().bytes_for_numel(1);
        let dst_off = off * row * esz;
        let n_bytes = src_c.layout.numel() * esz;
        if self.device().is_cuda() {
            let src_buf = src_c.storage.as_cuda().ok_or(SynaptixError::Unsupported("copy_rows_from: src"))?;
            let src_byte_off = src_c.layout.byte_offset();
            let stream = src_buf.stream().clone();
            let storage = Arc::get_mut(&mut self.storage)
                .ok_or_else(|| SynaptixError::Other("copy_rows_from: storage aliased".into()))?;
            let dst_buf = storage.as_cuda_mut().ok_or(SynaptixError::Unsupported("copy_rows_from: dst"))?;
            let src_sl = src_buf.slice().slice(src_byte_off..src_byte_off + n_bytes);
            let mut dst_sl = dst_buf.slice_mut().slice_mut(dst_off..dst_off + n_bytes);
            stream
                .memcpy_dtod(&src_sl, &mut dst_sl)
                .map_err(|e| SynaptixError::Cuda(format!("copy_rows_from dtod: {e:?}")))?;
            return Ok(());
        }
        let src_bytes = {
            let cpu = src_c
                .storage
                .as_cpu()
                .ok_or(SynaptixError::Unsupported("copy_rows_from: src не CPU"))?;
            let off = src_c.layout.byte_offset();
            cpu.as_bytes()[off..off + n_bytes].to_vec()
        };
        let storage = Arc::get_mut(&mut self.storage)
            .ok_or_else(|| SynaptixError::Other("copy_rows_from: storage aliased".into()))?;
        let dst = storage
            .as_cpu_mut()
            .ok_or(SynaptixError::Unsupported("copy_rows_from: dst не CPU"))?;
        dst.as_bytes_mut()[dst_off..dst_off + n_bytes].copy_from_slice(&src_bytes);
        Ok(())
    }

    /// Сырой device-указатель и длина (байт) региона плотного CUDA-тензора
    /// (база storage + byte-offset вьюхи). Для регион-копий слот-стриминга
    /// ([`crate::device::cuda::htod_into_region`]) — storage может быть разделён
    /// вьюхами (Arc::get_mut невозможен), порядок доступа упорядочивает
    /// вызывающий (события слота).
    pub fn cuda_region(&self) -> Result<(u64, usize)> {
        {
            if !self.layout.strides().is_contiguous(self.layout.shape()) {
                return Err(SynaptixError::NonContiguous);
            }
            let cb = self
                .storage
                .as_cuda()
                .ok_or(SynaptixError::Unsupported("cuda_region: не CUDA-тензор"))?;
            // SAFETY: home-стрим буфера → device_ptr без wait/record (same-stream
            // skip форка); указатель стабилен пока жив storage (Arc).
            let (base, _g) = {
                use cudarc::driver::DevicePtr;
                cb.slice().device_ptr(cb.stream())
            };
            let bytes = self.dtype().bytes_for_numel(self.layout.numel());
            return Ok((base + self.layout.byte_offset() as u64, bytes));
        }
    }

    /// In-place копия содержимого `src` в `self` (тот же shape/dtype/device).
    /// Указатель `self` не меняется (для стабильных CUDA-graph буферов: logits
    /// после forward кладутся в долгоживущий буфер). Требует уникального
    /// владения storage (`Arc::get_mut`).
    pub fn copy_from(&mut self, src: &Tensor) -> Result<()> {
        if self.device() != src.device() {
            return Err(SynaptixError::device_mismatch(self.device(), src.device()));
        }
        if self.dtype() != src.dtype() {
            return Err(SynaptixError::Unsupported("copy_from: dtype mismatch"));
        }
        if self.dims() != src.dims() {
            return Err(SynaptixError::Other(format!(
                "copy_from: shape mismatch self {:?} vs src {:?}",
                self.dims(),
                src.dims()
            )));
        }
        let src_c = if src.is_contiguous() { src.clone() } else { src.contiguous()? };
        let backend = registry::backend_for(self.device())?;
        let stream = Stream::default_for(self.device())?;
        let dst_layout = Layout::contiguous(self.shape().clone(), self.dtype());
        let storage = Arc::get_mut(&mut self.storage)
            .ok_or_else(|| SynaptixError::Other("copy_from: storage aliased".into()))?;
        backend.copy((&src_c.storage, &src_c.layout), (storage, &dst_layout), &stream)
    }

    /// In-place host→device запись `data` в U32-буфер `self` (без реаллокации —
    /// стабильный указатель для CUDA-graph replay: обновление токена/позиции
    /// перед `graph.launch`). `data.len()` должно совпадать с `self.numel()`.
    pub fn write_host_u32(&mut self, data: &[u32]) -> Result<()> {
        if self.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported("write_host_u32: tensor must be U32"));
        }
        if data.len() != self.numel() {
            return Err(SynaptixError::Other(format!(
                "write_host_u32: len {} != numel {}",
                data.len(),
                self.numel()
            )));
        }
        let storage = Arc::get_mut(&mut self.storage)
            .ok_or_else(|| SynaptixError::Other("write_host_u32: storage aliased".into()))?;
        match storage {
            Storage::Cpu(b) => {
                let bytes = b.as_bytes_mut();
                for (i, v) in data.iter().enumerate() {
                    bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                Ok(())
            }
            Storage::Cuda(b) => {
                let stream = b.stream().clone();
                let dst = b.slice_mut();
                let mut host_bytes = Vec::with_capacity(data.len() * 4);
                for v in data {
                    host_bytes.extend_from_slice(&v.to_le_bytes());
                }
                stream
                    .memcpy_htod(&host_bytes, dst)
                    .map_err(|e| SynaptixError::Cuda(format!("write_host_u32 memcpy_htod: {e:?}")))
            }
        }
    }

    /// Device-резидентный decode-шаг (T=1) GatedDeltaNet linear-attn (CUDA-graph).
    /// `self` — `qkv` (выход `in_proj_qkv`, F16 `[.., conv_dim]`); `a`/`b`/`z` —
    /// выходы `in_proj_{a,b,z}` (F16); `conv_w`/`norm_w` — веса (F16);
    /// `dt_bias`/`a_log` — F32; `conv_state` `[(K-1),conv_dim]` (F16) и `ssm_state`
    /// `[num_v,dk,dv]` (F32) обновляются in-place (стабильные указатели persistent-
    /// буфера → один граф для всех decode-шагов). Возвращает `out` F16
    /// `[1, value_dim]` (= нормированный gated SSM-выход до `out_proj`).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_attn_decode_step(
        &self,
        conv_w: &Tensor,
        a: &Tensor,
        b: &Tensor,
        dt_bias: &Tensor,
        a_log: &Tensor,
        z: &Tensor,
        norm_w: &Tensor,
        conv_state: &mut Tensor,
        ssm_state: &mut Tensor,
        num_k: usize,
        num_v: usize,
        dk: usize,
        dv: usize,
        conv_kernel: usize,
        q_scale: f32,
        eps: f32,
    ) -> Result<Self> {
        let dev = self.device();
        let value_dim = num_v * dv;
        let cc = |t: &Tensor| -> Result<Tensor> {
            if t.is_contiguous() { Ok(t.clone()) } else { t.contiguous() }
        };
        let qkv_c = cc(self)?;
        let conv_w_c = cc(conv_w)?;
        let a_c = cc(a)?;
        let b_c = cc(b)?;
        let dt_c = cc(dt_bias)?;
        let al_c = cc(a_log)?;
        let z_c = cc(z)?;
        let nw_c = cc(norm_w)?;
        if !conv_state.is_contiguous() || !ssm_state.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let cs_layout = conv_state.layout.clone();
        let ss_layout = ssm_state.layout.clone();

        let out_layout = Layout::contiguous(Shape::new(vec![1, value_dim]), DType::F16);
        let out_bytes = DType::F16.bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(dev)?;
        let stream = Stream::default_for(dev)?;
        let mut out_storage = backend.alloc_zeros(out_bytes, dev)?;
        let cs_storage = Arc::get_mut(&mut conv_state.storage)
            .ok_or_else(|| SynaptixError::Other("linear_attn_decode_step: conv_state aliased".into()))?;
        let ss_storage = Arc::get_mut(&mut ssm_state.storage)
            .ok_or_else(|| SynaptixError::Other("linear_attn_decode_step: ssm_state aliased".into()))?;
        backend.linear_attn_decode_step(
            (&qkv_c.storage, &qkv_c.layout),
            (&conv_w_c.storage, &conv_w_c.layout),
            (&a_c.storage, &a_c.layout),
            (&b_c.storage, &b_c.layout),
            (&dt_c.storage, &dt_c.layout),
            (&al_c.storage, &al_c.layout),
            (&z_c.storage, &z_c.layout),
            (&nw_c.storage, &nw_c.layout),
            (cs_storage, &cs_layout),
            (ss_storage, &ss_layout),
            (&mut out_storage, &out_layout),
            num_k,
            num_v,
            dk,
            dv,
            conv_kernel,
            q_scale,
            eps,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(out_storage), out_layout))
    }

    /// Полный chunked linear-attn prefill (T≥1) одним device-резидентным
    /// вызовом — замена host-mix-блока `LinearAttn::forward` (model.rs:879-915):
    /// causal_conv1d_chunk + silu + linear_attn_prep_scatter + chunk_gated_delta_rule.
    ///
    /// `self` = qkv `[1,T,conv_dim]` (compute_dtype F16/BF16/F32);
    /// `conv_w` `[conv_dim,K]` и `conv_state` `[(K-1),conv_dim]` — тот же dtype;
    /// `a`/`b` `[1,T,num_v]` — F16; `dt_bias`/`a_log` `[num_v]` — F32;
    /// `ssm_state` `[num_v,hk,hv]` — F32 in/out. Возвращает `out` `[num_v,T,hv]` F32.
    /// `T % chunk_size == 0` (caller паддит).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_attn_chunk_prefill(
        &self,
        conv_w: &Tensor,
        a: &Tensor,
        b: &Tensor,
        dt_bias: &Tensor,
        a_log: &Tensor,
        conv_state: &mut Tensor,
        ssm_state: &mut Tensor,
        num_k: usize,
        num_v: usize,
        hk: usize,
        hv: usize,
        conv_kernel: usize,
        chunk_size: usize,
        q_scale: f32,
        silu: bool,
    ) -> Result<Self> {
        let dev = self.device();
        let qd = self.dims();
        if qd.len() != 3 || qd[0] != 1 {
            return Err(SynaptixError::Other(format!(
                "linear_attn_chunk_prefill: qkv должен быть [1,T,conv_dim], получено {qd:?}"
            )));
        }
        let t_in = qd[1];
        let conv_dim = qd[2];
        let key_dim = num_k * hk;
        let value_dim = num_v * hv;
        if conv_dim != 2 * key_dim + value_dim {
            return Err(SynaptixError::Other(format!(
                "linear_attn_chunk_prefill: conv_dim={conv_dim} != 2*num_k*hk + num_v*hv \
                 (num_k={num_k} hk={hk} num_v={num_v} hv={hv})"
            )));
        }
        if t_in == 0 || chunk_size == 0 {
            return Err(SynaptixError::Other(format!(
                "linear_attn_chunk_prefill: T={t_in} chunk_size={chunk_size} invalid"
            )));
        }
        let t_pad = t_in.div_ceil(chunk_size) * chunk_size;
        let cc = |x: &Tensor| -> Result<Tensor> {
            if x.is_contiguous() { Ok(x.clone()) } else { x.contiguous() }
        };
        let qkv_c = cc(self)?;
        let cw_c = cc(conv_w)?;
        let a_c = cc(a)?;
        let b_c = cc(b)?;
        let dt_c = cc(dt_bias)?;
        let al_c = cc(a_log)?;
        if !conv_state.is_contiguous() || !ssm_state.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let cs_layout = conv_state.layout.clone();
        let ss_layout = ssm_state.layout.clone();

        // Out alloc'ится с t_pad (Backend op запускает scan на padded длине);
        // ниже narrow'им до t_in.
        let out_pad_layout =
            Layout::contiguous(Shape::new(vec![num_v, t_pad, hv]), DType::F32);
        let out_bytes = DType::F32.bytes_for_numel(out_pad_layout.numel());
        let backend = registry::backend_for(dev)?;
        let stream = Stream::default_for(dev)?;
        let mut out_storage = backend.alloc_zeros(out_bytes, dev)?;
        let cs_storage = Arc::get_mut(&mut conv_state.storage).ok_or_else(|| {
            SynaptixError::Other("linear_attn_chunk_prefill: conv_state aliased".into())
        })?;
        let ss_storage = Arc::get_mut(&mut ssm_state.storage).ok_or_else(|| {
            SynaptixError::Other("linear_attn_chunk_prefill: ssm_state aliased".into())
        })?;
        backend.linear_attn_chunk_prefill(
            (&qkv_c.storage, &qkv_c.layout),
            (&cw_c.storage, &cw_c.layout),
            (&a_c.storage, &a_c.layout),
            (&b_c.storage, &b_c.layout),
            (&dt_c.storage, &dt_c.layout),
            (&al_c.storage, &al_c.layout),
            (cs_storage, &cs_layout),
            (ss_storage, &ss_layout),
            (&mut out_storage, &out_pad_layout),
            num_k,
            num_v,
            hk,
            hv,
            conv_kernel,
            t_in,
            t_pad,
            chunk_size,
            q_scale,
            silu,
            &stream,
        )?;
        let out_pad = Tensor::from_parts(Arc::new(out_storage), out_pad_layout);
        if t_pad == t_in {
            Ok(out_pad)
        } else {
            out_pad.narrow(1, 0, t_in)?.contiguous()
        }
    }

    /// Chunked Gated-DeltaNet linear-attn prefill (T>1) на device — GPU-замена
    /// рекуррентного host-скана. `self` — `q` `[BH,T,HK]`; `k` `[BH,T,HK]`,
    /// `v` `[BH,T,HV]`, `g`/`beta` `[BH,T]` (F32; g = log-decay, beta post-sigmoid);
    /// `ssm_state` `[BH,HK,HV]` (F32) обновляется in-place. Возвращает `out`
    /// `[BH,T,HV]` (F32). q/k L2-нормализуются, q·=`q_scale`. Требует `T % chunk_size == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_prefill(
        &self,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        ssm_state: &mut Tensor,
        q_scale: f32,
        chunk_size: usize,
    ) -> Result<Self> {
        let dev = self.device();
        let qd = self.dims();
        if qd.len() != 3 {
            return Err(SynaptixError::Other(format!(
                "gated_delta_rule_prefill: q должен быть [BH,T,HK], получено {qd:?}"
            )));
        }
        let (bh, t, hk) = (qd[0], qd[1], qd[2]);
        let vd = v.dims();
        if vd.len() != 3 || vd[0] != bh || vd[1] != t {
            return Err(SynaptixError::Other(format!(
                "gated_delta_rule_prefill: v должен быть [{bh},{t},HV], получено {vd:?}"
            )));
        }
        let hv = vd[2];
        if chunk_size == 0 || t % chunk_size != 0 {
            return Err(SynaptixError::Other(format!(
                "gated_delta_rule_prefill: T={t} не кратно chunk_size={chunk_size}"
            )));
        }
        if ssm_state.dims() != [bh, hk, hv] {
            return Err(SynaptixError::Other(format!(
                "gated_delta_rule_prefill: ssm_state должен быть [{bh},{hk},{hv}], получено {:?}",
                ssm_state.dims()
            )));
        }
        let cc = |x: &Tensor| -> Result<Tensor> {
            if x.is_contiguous() { Ok(x.clone()) } else { x.contiguous() }
        };
        let qc = cc(self)?;
        let kc = cc(k)?;
        let vc = cc(v)?;
        let gc = cc(g)?;
        let bc = cc(beta)?;
        if !ssm_state.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let ss_layout = ssm_state.layout.clone();
        let out_layout = Layout::contiguous(Shape::new(vec![bh, t, hv]), DType::F32);
        let out_bytes = DType::F32.bytes_for_numel(out_layout.numel());
        let backend = registry::backend_for(dev)?;
        let stream = Stream::default_for(dev)?;
        let mut out_storage = backend.alloc_zeros(out_bytes, dev)?;
        let ss_storage = Arc::get_mut(&mut ssm_state.storage)
            .ok_or_else(|| SynaptixError::Other("gated_delta_rule_prefill: ssm_state aliased".into()))?;
        backend.gated_delta_rule_prefill(
            (&qc.storage, &qc.layout),
            (&kc.storage, &kc.layout),
            (&vc.storage, &vc.layout),
            (&gc.storage, &gc.layout),
            (&bc.storage, &bc.layout),
            (ss_storage, &ss_layout),
            (&mut out_storage, &out_layout),
            q_scale,
            bh,
            t,
            hk,
            hv,
            chunk_size,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(out_storage), out_layout))
    }
}
