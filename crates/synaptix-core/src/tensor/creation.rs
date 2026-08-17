use std::sync::Arc;

use crate::device::Device;
use crate::dtype::{DType, SynaptixScalar};
use crate::error::{Result, SynaptixError};
use crate::memory::arena;
use crate::tensor::Tensor;
use crate::tensor::layout::Layout;
use crate::tensor::shape::{IntoShape, Shape};
use crate::tensor::storage::{CpuBuf, Storage};

impl Tensor {
    pub fn zeros<S: IntoShape>(shape: S, dtype: DType, device: Device) -> Result<Self> {
        let shape = shape.into_shape();
        let n_bytes = dtype.bytes_for_numel(shape.numel());
        let storage = match device {
            Device::Cpu => Storage::Cpu(arena::alloc_zeros_cpu(n_bytes)),
            Device::Cuda(_) => {
                {
                    cuda_alloc_zeros(device, n_bytes)?
                }
            }
            _ => return Err(SynaptixError::Unsupported("device for Tensor::zeros")),
        };
        Ok(Tensor::from_parts(Arc::new(storage), Layout::contiguous(shape, dtype)))
    }

    pub fn from_vec<S: IntoShape, T: SynaptixScalar>(
        data: Vec<T>,
        shape: S,
        device: Device,
    ) -> Result<Self> {
        let shape = shape.into_shape();
        if shape.numel() != data.len() {
            return Err(SynaptixError::ShapeMismatch {
                expected: vec![data.len()],
                got: shape.dims().to_vec(),
            });
        }
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let storage = match device {
            Device::Cpu => Storage::Cpu(CpuBuf::from_vec(bytes)),
            Device::Cuda(_) => {
                {
                    cuda_alloc_from_bytes(device, &bytes)?
                }
            }
            _ => return Err(SynaptixError::Unsupported("device for Tensor::from_vec")),
        };
        let layout = Layout::contiguous(shape, T::DTYPE);
        Ok(Tensor::from_parts(Arc::new(storage), layout))
    }

    pub fn from_slice<S: IntoShape, T: SynaptixScalar>(
        data: &[T],
        shape: S,
        device: Device,
    ) -> Result<Self> {
        Tensor::from_vec(data.to_vec(), shape, device)
    }

    pub fn from_raw_bytes<S: IntoShape>(
        bytes: Vec<u8>,
        shape: S,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let shape = shape.into_shape();
        let expected = dtype.bytes_for_numel(shape.numel());
        if bytes.len() != expected {
            return Err(SynaptixError::Other(format!(
                "from_raw_bytes: ожидалось {} байт для shape={:?} dtype={:?}, получено {}",
                expected,
                shape.dims(),
                dtype,
                bytes.len()
            )));
        }
        let storage = match device {
            Device::Cpu => Storage::Cpu(CpuBuf::from_vec(bytes)),
            Device::Cuda(_) => {
                {
                    cuda_alloc_from_bytes(device, &bytes)?
                }
            }
            _ => return Err(SynaptixError::Unsupported("device for Tensor::from_raw_bytes")),
        };
        Ok(Tensor::from_parts(Arc::new(storage), Layout::contiguous(shape, dtype)))
    }

    /// Как [`from_raw_bytes`], но принимает заимствованный слайс. На CUDA это
    /// заливает H2D напрямую из слайса (например, mmap-региона bundle'а) без
    /// промежуточного владеющего `Vec` — экономит одну host-копию всего веса.
    /// На CPU слайс копируется в owned `CpuBuf` (копия неизбежна).
    pub fn from_raw_slice<S: IntoShape>(
        bytes: &[u8],
        shape: S,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let shape = shape.into_shape();
        let expected = dtype.bytes_for_numel(shape.numel());
        if bytes.len() != expected {
            return Err(SynaptixError::Other(format!(
                "from_raw_slice: ожидалось {} байт для shape={:?} dtype={:?}, получено {}",
                expected,
                shape.dims(),
                dtype,
                bytes.len()
            )));
        }
        let storage = match device {
            Device::Cpu => Storage::Cpu(CpuBuf::from_vec(bytes.to_vec())),
            Device::Cuda(_) => {
                {
                    cuda_alloc_from_bytes(device, bytes)?
                }
            }
            _ => return Err(SynaptixError::Unsupported("device for Tensor::from_raw_slice")),
        };
        Ok(Tensor::from_parts(Arc::new(storage), Layout::contiguous(shape, dtype)))
    }

    pub fn ones<S: IntoShape>(shape: S, dtype: DType, device: Device) -> Result<Self> {
        Tensor::ones_with_device(shape.into_shape(), dtype, device)
    }

    pub fn ones_like(&self) -> Result<Self> {
        Tensor::ones_with_device(self.shape().clone(), self.dtype(), self.device())
    }

    pub fn zeros_like(&self) -> Result<Self> {
        Tensor::zeros(self.shape().clone(), self.dtype(), self.device())
    }

    fn ones_with_device(shape: Shape, dtype: DType, device: Device) -> Result<Self> {
        let numel = shape.numel();
        let bytes = fill_bytes_one(dtype, numel)?;
        let storage = match device {
            Device::Cpu => Storage::Cpu(CpuBuf::from_vec(bytes)),
            Device::Cuda(_) => {
                {
                    cuda_alloc_from_bytes(device, &bytes)?
                }
            }
            _ => return Err(SynaptixError::Unsupported("device for Tensor::ones")),
        };
        Ok(Tensor::from_parts(Arc::new(storage), Layout::contiguous(shape, dtype)))
    }

    pub fn arange<T>(start: T, end: T, device: Device) -> Result<Self>
    where
        T: SynaptixScalar + num_traits::NumOps + PartialOrd + Copy + num_traits::One + num_traits::Zero,
    {
        let mut data = Vec::new();
        let mut v = start;
        while v < end {
            data.push(v);
            v = v + T::one();
        }
        let n = data.len();
        Tensor::from_vec(data, (n,), device)
    }

    pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(SynaptixError::Unsupported("cat: empty list"));
        }
        let first = tensors[0];
        let rank = first.rank();
        if dim >= rank {
            return Err(SynaptixError::DimOutOfRange { dim, rank });
        }
        let dtype = first.dtype();
        let device = first.device();
        for t in tensors.iter().skip(1) {
            if t.dtype() != dtype {
                return Err(SynaptixError::dtype_mismatch(dtype, t.dtype()));
            }
            if t.device() != device {
                return Err(SynaptixError::device_mismatch(device, t.device()));
            }
            if t.rank() != rank {
                return Err(SynaptixError::RankMismatch { expected: rank, got: t.rank() });
            }
            for (i, (&a, &b)) in first.dims().iter().zip(t.dims()).enumerate() {
                if i != dim && a != b {
                    return Err(SynaptixError::shape_mismatch(first.dims(), t.dims()));
                }
            }
        }
        if !device.is_cpu() {
            {
                return cat_cuda(tensors, dim);
            }
        }
        let mut out_dims = first.dims().to_vec();
        out_dims[dim] = tensors.iter().map(|t| t.dims()[dim]).sum();
        let out_shape = Shape::new(out_dims.clone());
        let elem_bytes = dtype.size_in_bits() / 8;
        let total_bytes = out_shape.numel() * elem_bytes;
        let mut out_bytes = vec![0u8; total_bytes];

        let outer: usize = first.dims()[..dim].iter().product();
        let inner_first: usize = first.dims()[dim + 1..].iter().product();
        let inner_bytes_each = |t: &Tensor| t.dims()[dim] * inner_first * elem_bytes;
        let out_slice_bytes_per_outer: usize = out_dims[dim] * inner_first * elem_bytes;

        for o in 0..outer {
            let mut written = 0usize;
            for t in tensors {
                if !t.is_contiguous() {
                    return Err(SynaptixError::NonContiguous);
                }
                let src = match &*t.storage {
                    Storage::Cpu(b) => b.as_bytes(),
                    _ => return Err(SynaptixError::Unsupported("cat: non-cpu storage")),
                };
                let chunk = inner_bytes_each(t);
                let src_start = (o * chunk + t.layout.offset() * elem_bytes) as usize;
                let src_end = src_start + chunk;
                let dst_start = o * out_slice_bytes_per_outer + written;
                let dst_end = dst_start + chunk;
                out_bytes[dst_start..dst_end].copy_from_slice(&src[src_start..src_end]);
                written += chunk;
            }
        }
        let storage = Storage::Cpu(CpuBuf::from_vec(out_bytes));
        let mut out = Tensor::from_parts(Arc::new(storage), Layout::contiguous(out_shape, dtype));
        crate::grad::try_attach_grad_fn(
            crate::grad::GradOp::Cat { inputs: tensors.to_vec(), dim },
            &mut out,
        )?;
        Ok(out)
    }

    pub fn stack(tensors: &[&Tensor], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(SynaptixError::Unsupported("stack: empty list"));
        }
        let unsqueezed: Vec<_> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        Tensor::cat(&refs, dim)
    }
}

fn fill_bytes_one(dtype: DType, numel: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; dtype.bytes_for_numel(numel)];
    match dtype {
        DType::F32 => {
            let s: &mut [f32] = bytemuck::cast_slice_mut(&mut out);
            s.fill(1.0);
        }
        DType::F64 => {
            let s: &mut [f64] = bytemuck::cast_slice_mut(&mut out);
            s.fill(1.0);
        }
        DType::F16 => {
            let s: &mut [half::f16] = bytemuck::cast_slice_mut(&mut out);
            s.fill(half::f16::ONE);
        }
        DType::BF16 => {
            let s: &mut [half::bf16] = bytemuck::cast_slice_mut(&mut out);
            s.fill(half::bf16::ONE);
        }
        DType::U8 => out.fill(1),
        DType::U32 => {
            let s: &mut [u32] = bytemuck::cast_slice_mut(&mut out);
            s.fill(1);
        }
        DType::I32 => {
            let s: &mut [i32] = bytemuck::cast_slice_mut(&mut out);
            s.fill(1);
        }
        DType::I64 => {
            let s: &mut [i64] = bytemuck::cast_slice_mut(&mut out);
            s.fill(1);
        }
        _ => return Err(SynaptixError::Unsupported("ones on quantized dtype")),
    }
    Ok(out)
}

/// `Tensor::cat` на CUDA через device-to-device memcpy чанков (без CUDA C kernel).
/// Зеркалит CPU-логику: для каждого outer-слайса последовательно копирует
/// contiguous-чанки каждого входа в выход. Используется в KV-cache append (dim=2)
/// и apply_rope (dim=3).
fn cat_cuda(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    let first = tensors[0];
    let dtype = first.dtype();
    let device = first.device();
    let ord = device.ordinal();
    let elem_bytes = dtype.size_in_bits() / 8;

    let mut out_dims = first.dims().to_vec();
    out_dims[dim] = tensors.iter().map(|t| t.dims()[dim]).sum();
    let out_shape = Shape::new(out_dims.clone());
    let total_bytes = out_shape.numel() * elem_bytes;

    let outer: usize = first.dims()[..dim].iter().product();
    let inner_first: usize = first.dims()[dim + 1..].iter().product();
    let out_slice_per_outer: usize = out_dims[dim] * inner_first * elem_bytes;

    let stream = crate::device::cuda::default_stream(ord)?;
    let mut out_storage = cuda_alloc_zeros(device, total_bytes)?;
    {
        use cudarc::driver::{sys, DevicePtr, DevicePtrMut};
        let out_buf = out_storage
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cat_cuda: out storage non-cuda"))?;
        // Базовый device-ptr выходного буфера (контигуальный).
        let dst_base: sys::CUdeviceptr = {
            let (p, _g) = out_buf.slice_mut().device_ptr_mut(&stream);
            p
        };
        let cu_stream = stream.cu_stream();
        // Одна 2D-копия на входной тензор: вместо `outer` микро-memcpy
        // (storm из тысяч launch'ей при cat по внутренней оси, где outer огромен)
        // — единый cuMemcpy2D с src/dst pitch. Height = outer строк, ширина =
        // блок тензора по dim. Сводит O(outer·N) launch'ей к O(N).
        let mut written = 0usize;
        for t in tensors {
            if !t.is_contiguous() {
                return Err(SynaptixError::NonContiguous);
            }
            let src_buf = t
                .storage()
                .as_cuda()
                .ok_or(SynaptixError::Unsupported("cat_cuda: src storage non-cuda"))?;
            let chunk = t.dims()[dim] * inner_first * elem_bytes;
            let src_base: sys::CUdeviceptr = {
                let (p, _g) = src_buf.slice().device_ptr(&stream);
                p
            };
            let copy = sys::CUDA_MEMCPY2D_st {
                srcXInBytes: 0,
                srcY: 0,
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: std::ptr::null(),
                srcDevice: src_base + t.layout().byte_offset() as sys::CUdeviceptr,
                srcArray: std::ptr::null_mut(),
                srcPitch: chunk,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: std::ptr::null_mut(),
                dstDevice: dst_base + written as sys::CUdeviceptr,
                dstArray: std::ptr::null_mut(),
                dstPitch: out_slice_per_outer,
                WidthInBytes: chunk,
                Height: outer,
            };
            unsafe {
                sys::cuMemcpy2DAsync_v2(&copy, cu_stream)
                    .result()
                    .map_err(|e| SynaptixError::Cuda(format!("cat cuMemcpy2D: {e:?}")))?;
            }
            written += chunk;
        }
    }
    let mut out = Tensor::from_parts(Arc::new(out_storage), Layout::contiguous(out_shape, dtype));
    crate::grad::try_attach_grad_fn(
        crate::grad::GradOp::Cat { inputs: tensors.to_vec(), dim },
        &mut out,
    )?;
    Ok(out)
}

pub(crate) fn cuda_alloc_zeros(device: Device, n_bytes: usize) -> Result<Storage> {
    let ord = device.ordinal();
    let stream = crate::device::cuda::alloc_stream(ord)?;
    let buf = match crate::device::cuda::alloc_act_zeros::<u8>(&stream, n_bytes) {
        Ok(b) => b,
        Err(_) => {
            // OOM: пул async-аллокатора держит освобождённые блоки (фрагментация)
            // — вернуть их драйверу (trim) и повторить, как CudaBackend::alloc_*.
            // Сперва sync ВСЕХ стримов: cuMemFreeAsync-освобождения исполняются
            // в порядке СВОЕГО стрима, до sync trim их не видит.
            let _ = crate::device::cuda::synchronize_all(ord);
            let _ = crate::memory::cuda_pool::trim_pools_on_oom(ord);
            {
                // ретраи с эскалацией — фрагментация транзиентна (см. CudaBackend)
                let mut got = None;
                for attempt in 0..5u32 {
                    std::thread::sleep(std::time::Duration::from_millis(50 * attempt as u64));
                    let _ = crate::device::cuda::synchronize_all(ord);
                    let _ = crate::memory::cuda_pool::trim_pools_on_oom(ord);
                    if let Ok(b) = crate::device::cuda::alloc_act_zeros::<u8>(&stream, n_bytes) {
                        got = Some(b);
                        break;
                    }
                }
                got.ok_or_else(|| {
                    for (bytes, count) in crate::memory::cuda_pool::live_alloc_top(12) {
                        eprintln!("[OOM_TOP] {bytes:>12} B × {count} = {:.2}GB",
                            bytes as f64 * count as f64 / 1e9);
                    }
                    let (free, total) = crate::device::cuda::mem_info(ord).unwrap_or((0, 0));
                    let (rsv, used) =
                        crate::memory::cuda_pool::cuda_mempool_stats(ord).unwrap_or((0, 0));
                    eprintln!(
                        "[OOM_SUM] live(наш учёт)={:.2}GB free={:.2}GB total={:.2}GB pool_reserved={:.2}GB pool_used={:.2}GB",
                        crate::memory::cuda_pool::cuda_allocated_bytes() as f64 / 1e9,
                        free as f64 / 1e9, total as f64 / 1e9,
                        rsv as f64 / 1e9, used as f64 / 1e9
                    );
                    eprintln!("[OOM_BT] cat/zeros alloc_zeros({n_bytes}):\n{}",
                        std::backtrace::Backtrace::force_capture());
                    SynaptixError::Cuda(format!("alloc_zeros({n_bytes}) after trim+retries: OOM"))
                })?
            }
        }
    };
    let ctx = crate::device::cuda::get(ord)?;
    Ok(Storage::Cuda(crate::tensor::storage::CudaBuf::new(ctx, stream, buf, ord)))
}

pub(crate) fn cuda_alloc_from_bytes(device: Device, bytes: &[u8]) -> Result<Storage> {
    let ord = device.ordinal();
    let stream = crate::device::cuda::alloc_stream(ord)?;
    // offload-загрузка: байты внутри pinned-кэша ckpt → DMA из персистентной
    // pinned-копии (async на stream — потребитель синкает stream); иначе pinned
    // staging (45 GB/s vs 3.6 pageable).
    let buf = if let Some(r) = crate::device::cuda::offload_pin_cache_htod(&stream, bytes) {
        r?
    } else if let Some(r) = crate::device::cuda::pin_mirror_htod(&stream, bytes) {
        r?
    } else if crate::device::cuda::offload_pinned_enabled() && !bytes.is_empty() {
        crate::device::cuda::pinned_htod(&stream, bytes)?
    } else {
        // Как `clone_htod`, но аллокация — из staging-пула (см.
        // `cuda::alloc_bytes_uninit`): байты весов не должны оседать в пуле
        // рядом с готовыми квантованными весами.
        let mut dst = unsafe { crate::device::cuda::alloc_bytes_uninit(&stream, bytes.len()) }
            .map_err(|e| SynaptixError::Cuda(format!("alloc H2D({} bytes): {e:?}", bytes.len())))?;
        stream
            .memcpy_htod(bytes, &mut dst)
            .map_err(|e| SynaptixError::Cuda(format!("memcpy_htod({} bytes): {e:?}", bytes.len())))?;
        dst
    };
    let ctx = crate::device::cuda::get(ord)?;
    Ok(Storage::Cuda(crate::tensor::storage::CudaBuf::new(ctx, stream, buf, ord)))
}
