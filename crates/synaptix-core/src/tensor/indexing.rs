use std::sync::Arc;

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;
use crate::tensor::layout::Layout;
use crate::tensor::shape::Shape;
use crate::tensor::storage::{CpuBuf, Storage};

impl Tensor {
    pub fn gather(&self, indices: &Tensor, dim: usize) -> Result<Self> {
        require_cpu(self, "gather")?;
        require_cpu(indices, "gather indices")?;
        if self.device() != indices.device() {
            return Err(SynaptixError::device_mismatch(self.device(), indices.device()));
        }
        let rank = self.rank();
        if dim >= rank {
            return Err(SynaptixError::DimOutOfRange { dim, rank });
        }
        if indices.rank() != rank {
            return Err(SynaptixError::RankMismatch {
                expected: rank,
                got: indices.rank(),
            });
        }
        for k in 0..rank {
            if k == dim {
                continue;
            }
            if indices.dims()[k] != self.dims()[k] {
                return Err(SynaptixError::shape_mismatch(self.dims(), indices.dims()));
            }
        }
        let src = self.contiguous()?;
        let idx = indices.contiguous()?;
        let out_layout = Layout::contiguous(Shape::new(indices.dims().to_vec()), src.dtype());
        let mut out_bytes = vec![0u8; src.dtype().bytes_for_numel(out_layout.numel())];
        let elem = elem_size(src.dtype())?;
        let src_bytes = cpu_bytes(&src)?;
        let src_dim = src.dims()[dim];
        let idx_dims = idx.dims().to_vec();
        let outer: usize = idx_dims[..dim].iter().product();
        let inner: usize = idx_dims[dim + 1..].iter().product();
        let idx_len = idx_dims[dim];
        let src_inner_stride = src.dims()[dim + 1..].iter().product::<usize>();
        let read_index = |buf: &[u8], off: usize| -> Result<usize> {
            match idx.dtype() {
                DType::U32 => Ok(read_u32(buf, off) as usize),
                DType::I64 => Ok(read_i64(buf, off) as usize),
                DType::I32 => Ok(read_i32(buf, off) as usize),
                _ => Err(SynaptixError::Unsupported("gather: indices dtype")),
            }
        };
        let idx_bytes = cpu_bytes(&idx)?;
        let idx_elem = elem_size(idx.dtype())?;
        for o in 0..outer {
            for k in 0..idx_len {
                for i in 0..inner {
                    let pos = ((o * idx_len + k) * inner + i) * idx_elem;
                    let raw = read_index(idx_bytes, pos)?;
                    if raw >= src_dim {
                        return Err(SynaptixError::Other(format!(
                            "gather: index {raw} out of range for dim {dim} size {src_dim}"
                        )));
                    }
                    let src_pos = ((o * src_dim + raw) * src_inner_stride + i) * elem;
                    let dst_pos = ((o * idx_len + k) * inner + i) * elem;
                    out_bytes[dst_pos..dst_pos + elem]
                        .copy_from_slice(&src_bytes[src_pos..src_pos + elem]);
                }
            }
        }
        let storage = Storage::Cpu(CpuBuf::from_vec(out_bytes));
        let mut out = Tensor::from_parts(Arc::new(storage), out_layout);
        crate::grad::try_attach_grad_fn(
            crate::grad::GradOp::Gather { input: self, indices, dim },
            &mut out,
        )?;
        Ok(out)
    }

    pub fn scatter_add(&self, dim: usize, indices: &Tensor, values: &Tensor) -> Result<Self> {
        require_cpu(self, "scatter_add")?;
        require_cpu(indices, "scatter_add indices")?;
        require_cpu(values, "scatter_add values")?;
        if self.device() != indices.device() || self.device() != values.device() {
            return Err(SynaptixError::device_mismatch(self.device(), indices.device()));
        }
        if self.dtype() != values.dtype() {
            return Err(SynaptixError::dtype_mismatch(self.dtype(), values.dtype()));
        }
        let rank = self.rank();
        if dim >= rank {
            return Err(SynaptixError::DimOutOfRange { dim, rank });
        }
        if indices.rank() != rank || values.rank() != rank {
            return Err(SynaptixError::RankMismatch {
                expected: rank,
                got: indices.rank(),
            });
        }
        if indices.dims() != values.dims() {
            return Err(SynaptixError::shape_mismatch(values.dims(), indices.dims()));
        }
        for k in 0..rank {
            if k == dim {
                continue;
            }
            if indices.dims()[k] != self.dims()[k] {
                return Err(SynaptixError::shape_mismatch(self.dims(), indices.dims()));
            }
        }
        let src = self.contiguous()?;
        let idx = indices.contiguous()?;
        let val = values.contiguous()?;
        let elem = elem_size(src.dtype())?;
        let mut out_bytes = cpu_bytes(&src)?.to_vec();
        let val_bytes = cpu_bytes(&val)?;
        let idx_bytes = cpu_bytes(&idx)?;
        let idx_elem = elem_size(idx.dtype())?;
        let src_dim = src.dims()[dim];
        let val_dims = val.dims().to_vec();
        let outer: usize = val_dims[..dim].iter().product();
        let inner: usize = val_dims[dim + 1..].iter().product();
        let val_dim_len = val_dims[dim];
        for o in 0..outer {
            for k in 0..val_dim_len {
                for i in 0..inner {
                    let idx_pos = ((o * val_dim_len + k) * inner + i) * idx_elem;
                    let target = match idx.dtype() {
                        DType::U32 => read_u32(idx_bytes, idx_pos) as usize,
                        DType::I64 => read_i64(idx_bytes, idx_pos) as usize,
                        DType::I32 => read_i32(idx_bytes, idx_pos) as usize,
                        _ => return Err(SynaptixError::Unsupported("scatter_add: indices dtype")),
                    };
                    if target >= src_dim {
                        return Err(SynaptixError::Other(format!(
                            "scatter_add: index {target} out of range for dim {dim} size {src_dim}"
                        )));
                    }
                    let val_pos = ((o * val_dim_len + k) * inner + i) * elem;
                    let dst_pos = ((o * src_dim + target) * inner + i) * elem;
                    add_in_place(
                        &mut out_bytes[dst_pos..dst_pos + elem],
                        &val_bytes[val_pos..val_pos + elem],
                        src.dtype(),
                    )?;
                }
            }
        }
        let storage = Storage::Cpu(CpuBuf::from_vec(out_bytes));
        let layout = Layout::contiguous(Shape::new(src.dims().to_vec()), src.dtype());
        Ok(Tensor::from_parts(Arc::new(storage), layout))
    }

    pub fn index_select(&self, dim: usize, indices: &Tensor) -> Result<Self> {
        if self.device() != indices.device() {
            return Err(SynaptixError::device_mismatch(self.device(), indices.device()));
        }
        let rank = self.rank();
        if dim >= rank {
            return Err(SynaptixError::DimOutOfRange { dim, rank });
        }
        if !self.device().is_cpu() {
            #[cfg(feature = "cuda")]
            {
                return self.index_select_cuda(dim, indices);
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(SynaptixError::Unsupported(
                    "index_select on non-CPU device requires cuda feature",
                ));
            }
        }
        let src = self.contiguous()?;
        let idx = indices.contiguous()?;
        let idx_numel = idx.numel();
        let idx_out_shape: Vec<usize> = idx.dims().to_vec();
        let combined_rank = dim + idx_out_shape.len() + (rank - dim - 1);
        let mut combined_dims = Vec::with_capacity(combined_rank);
        combined_dims.extend_from_slice(&src.dims()[..dim]);
        combined_dims.extend_from_slice(&idx_out_shape);
        combined_dims.extend_from_slice(&src.dims()[dim + 1..]);
        let out_shape = Shape::new(combined_dims);
        let out_layout = Layout::contiguous(out_shape.clone(), src.dtype());
        let elem = elem_size(src.dtype())?;
        let outer: usize = src.dims()[..dim].iter().product();
        let inner: usize = src.dims()[dim + 1..].iter().product();
        let src_dim_size = src.dims()[dim];
        let mut out_bytes = vec![0u8; src.dtype().bytes_for_numel(out_layout.numel())];
        let src_bytes = cpu_bytes(&src)?;
        let idx_bytes = cpu_bytes(&idx)?;
        let idx_elem = elem_size(idx.dtype())?;
        let read_index = |buf: &[u8], off: usize| -> Result<usize> {
            match idx.dtype() {
                DType::U32 => Ok(read_u32(buf, off) as usize),
                DType::I64 => Ok(read_i64(buf, off) as usize),
                DType::I32 => Ok(read_i32(buf, off) as usize),
                _ => Err(SynaptixError::Unsupported("index_select: indices dtype")),
            }
        };
        for o in 0..outer {
            for k in 0..idx_numel {
                let raw = read_index(idx_bytes, k * idx_elem)?;
                if raw >= src_dim_size {
                    return Err(SynaptixError::Other(format!(
                        "index_select: index {raw} out of range for dim {dim} size {src_dim_size}"
                    )));
                }
                let src_pos = ((o * src_dim_size + raw) * inner) * elem;
                let dst_pos = ((o * idx_numel + k) * inner) * elem;
                let chunk = inner * elem;
                out_bytes[dst_pos..dst_pos + chunk]
                    .copy_from_slice(&src_bytes[src_pos..src_pos + chunk]);
            }
        }
        let storage = Storage::Cpu(CpuBuf::from_vec(out_bytes));
        let mut out = Tensor::from_parts(Arc::new(storage), out_layout);
        crate::grad::try_attach_grad_fn(
            crate::grad::GradOp::IndexSelect { input: self, indices, dim },
            &mut out,
        )?;
        Ok(out)
    }

    /// `index_select` на CUDA через device-to-device memcpy. Индексы малы
    /// (`positions`/`input_ids`) и читаются на host (`clone_dtoh`); затем для
    /// каждого выбранного индекса копируется contiguous-блок `inner` элементов.
    /// Используется в `token_embedding` (dim=0) и `RopeCache::select_positions`.
    #[cfg(feature = "cuda")]
    fn index_select_cuda(&self, dim: usize, indices: &Tensor) -> Result<Self> {
        let src = self.contiguous()?;
        let idx = indices.contiguous()?;
        let device = src.device();
        let ord = device.ordinal();
        let elem = elem_size(src.dtype())?;

        let idx_numel = idx.numel();
        let idx_out_shape: Vec<usize> = idx.dims().to_vec();
        let mut combined_dims = Vec::with_capacity(dim + idx_out_shape.len() + (self.rank() - dim - 1));
        combined_dims.extend_from_slice(&src.dims()[..dim]);
        combined_dims.extend_from_slice(&idx_out_shape);
        combined_dims.extend_from_slice(&src.dims()[dim + 1..]);
        let out_shape = Shape::new(combined_dims);
        let out_layout = Layout::contiguous(out_shape.clone(), src.dtype());

        let outer: usize = src.dims()[..dim].iter().product();
        let inner: usize = src.dims()[dim + 1..].iter().product();
        let src_dim_size = src.dims()[dim];

        let stream = crate::device::cuda::default_stream(ord)?;

        // Индексы → host.
        let idx_buf = idx
            .storage()
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("index_select_cuda: idx non-cuda"))?;
        let idx_bytes_full = stream
            .clone_dtoh(idx_buf.slice())
            .map_err(|e| SynaptixError::Cuda(format!("index_select clone_dtoh: {e:?}")))?;
        let idx_bytes = &idx_bytes_full[idx.layout().byte_offset()..];
        let idx_elem = elem_size(idx.dtype())?;
        let read_index = |off: usize| -> Result<usize> {
            match idx.dtype() {
                DType::U32 => Ok(read_u32(idx_bytes, off) as usize),
                DType::I64 => Ok(read_i64(idx_bytes, off) as usize),
                DType::I32 => Ok(read_i32(idx_bytes, off) as usize),
                _ => Err(SynaptixError::Unsupported("index_select: indices dtype")),
            }
        };

        let total_bytes = src.dtype().bytes_for_numel(out_layout.numel());
        let mut out_storage = crate::tensor::creation::cuda_alloc_zeros(device, total_bytes)?;
        let src_buf = src
            .storage()
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("index_select_cuda: src non-cuda"))?;
        let src_byte_off = src.layout().byte_offset();
        let chunk = inner * elem;

        // Индексы на host один раз + проверка границ.
        let idx_vec: Vec<usize> = (0..idx_numel)
            .map(|k| read_index(k * idx_elem))
            .collect::<Result<Vec<_>>>()?;
        for &r in &idx_vec {
            if r >= src_dim_size {
                return Err(SynaptixError::Other(format!(
                    "index_select: index {r} out of range for dim {dim} size {src_dim_size}"
                )));
            }
        }
        // Смежный возрастающий диапазон (rope positions [s..s+n]) → один memcpy на
        // outer вместо N построчных. Критично для prefill: иначе rope-select даёт
        // O(S) memcpy-launch'ей на КАЖДЫЙ из ~112 apply_rope-вызовов.
        let contiguous_range =
            idx_numel > 0 && idx_vec.iter().enumerate().all(|(k, &r)| r == idx_vec[0] + k);
        {
            let out_buf = out_storage
                .as_cuda_mut()
                .ok_or(SynaptixError::Unsupported("index_select_cuda: out non-cuda"))?;
            if contiguous_range {
                let base = idx_vec[0];
                let block = idx_numel * inner * elem;
                for o in 0..outer {
                    let src_start = src_byte_off + ((o * src_dim_size + base) * inner) * elem;
                    let dst_start = (o * idx_numel * inner) * elem;
                    let src_view = src_buf.slice().slice(src_start..src_start + block);
                    let mut dst_view = out_buf.slice_mut().slice_mut(dst_start..dst_start + block);
                    stream.memcpy_dtod(&src_view, &mut dst_view).map_err(|e| {
                        SynaptixError::Cuda(format!("index_select memcpy_dtod: {e:?}"))
                    })?;
                }
            } else {
                for o in 0..outer {
                    for (k, &raw) in idx_vec.iter().enumerate() {
                        let src_start = src_byte_off + ((o * src_dim_size + raw) * inner) * elem;
                        let dst_start = ((o * idx_numel + k) * inner) * elem;
                        let src_view = src_buf.slice().slice(src_start..src_start + chunk);
                        let mut dst_view =
                            out_buf.slice_mut().slice_mut(dst_start..dst_start + chunk);
                        stream.memcpy_dtod(&src_view, &mut dst_view).map_err(|e| {
                            SynaptixError::Cuda(format!("index_select memcpy_dtod: {e:?}"))
                        })?;
                    }
                }
            }
        }
        let mut out = Tensor::from_parts(Arc::new(out_storage), out_layout);
        crate::grad::try_attach_grad_fn(
            crate::grad::GradOp::IndexSelect { input: self, indices, dim },
            &mut out,
        )?;
        Ok(out)
    }

    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Self> {
        require_cpu(self, "masked_fill")?;
        require_cpu(mask, "masked_fill mask")?;
        if mask.dtype() != DType::U8 && mask.dtype() != DType::U32 && !mask.dtype().is_float() {
            return Err(SynaptixError::Unsupported(
                "masked_fill: mask dtype (use bool-like u8/u32 or float)",
            ));
        }
        let target_shape = crate::tensor::broadcast::broadcast_shape(self.dims(), mask.dims())?;
        let src = self.broadcast_as(target_shape.clone())?.contiguous()?;
        let m = mask.broadcast_as(target_shape.clone())?.contiguous()?;
        let elem = elem_size(src.dtype())?;
        let mut out_bytes = cpu_bytes(&src)?.to_vec();
        let m_bytes = cpu_bytes(&m)?;
        let m_elem = elem_size(m.dtype())?;
        let n = src.numel();
        let value_bytes = scalar_bytes(value, src.dtype())?;
        for i in 0..n {
            let m_pos = i * m_elem;
            let is_true = match m.dtype() {
                DType::U8 => m_bytes[m_pos] != 0,
                DType::U32 => read_u32(m_bytes, m_pos) != 0,
                DType::F32 => read_f32(m_bytes, m_pos) != 0.0,
                DType::F16 => half::f16::from_le_bytes([m_bytes[m_pos], m_bytes[m_pos + 1]])
                    .to_f32()
                    != 0.0,
                DType::BF16 => half::bf16::from_le_bytes([m_bytes[m_pos], m_bytes[m_pos + 1]])
                    .to_f32()
                    != 0.0,
                DType::F64 => read_f64(m_bytes, m_pos) != 0.0,
                _ => return Err(SynaptixError::Unsupported("masked_fill: mask dtype")),
            };
            if is_true {
                let dst_pos = i * elem;
                out_bytes[dst_pos..dst_pos + elem].copy_from_slice(&value_bytes);
            }
        }
        let storage = Storage::Cpu(CpuBuf::from_vec(out_bytes));
        let layout = Layout::contiguous(target_shape, src.dtype());
        let mut out = Tensor::from_parts(Arc::new(storage), layout);
        crate::grad::try_attach_grad_fn(
            crate::grad::GradOp::MaskedFill { input: self, mask, value },
            &mut out,
        )?;
        Ok(out)
    }

    pub fn where_cond(cond: &Tensor, a: &Tensor, b: &Tensor) -> Result<Self> {
        require_cpu(cond, "where cond")?;
        require_cpu(a, "where a")?;
        require_cpu(b, "where b")?;
        if a.dtype() != b.dtype() {
            return Err(SynaptixError::dtype_mismatch(a.dtype(), b.dtype()));
        }
        if a.device() != b.device() || a.device() != cond.device() {
            return Err(SynaptixError::device_mismatch(a.device(), b.device()));
        }
        let shape_ab = crate::tensor::broadcast::broadcast_shape(a.dims(), b.dims())?;
        let shape = crate::tensor::broadcast::broadcast_shape(cond.dims(), shape_ab.dims())?;
        let aa = a.broadcast_as(shape.clone())?.contiguous()?;
        let bb = b.broadcast_as(shape.clone())?.contiguous()?;
        let cc = cond.broadcast_as(shape.clone())?.contiguous()?;
        let elem = elem_size(aa.dtype())?;
        let n = aa.numel();
        let mut out_bytes = vec![0u8; aa.dtype().bytes_for_numel(n)];
        let a_bytes = cpu_bytes(&aa)?;
        let b_bytes = cpu_bytes(&bb)?;
        let c_bytes = cpu_bytes(&cc)?;
        let c_elem = elem_size(cc.dtype())?;
        for i in 0..n {
            let c_pos = i * c_elem;
            let is_true = match cc.dtype() {
                DType::U8 => c_bytes[c_pos] != 0,
                DType::U32 => read_u32(c_bytes, c_pos) != 0,
                DType::F32 => read_f32(c_bytes, c_pos) != 0.0,
                DType::F16 => half::f16::from_le_bytes([c_bytes[c_pos], c_bytes[c_pos + 1]])
                    .to_f32()
                    != 0.0,
                DType::BF16 => half::bf16::from_le_bytes([c_bytes[c_pos], c_bytes[c_pos + 1]])
                    .to_f32()
                    != 0.0,
                DType::F64 => read_f64(c_bytes, c_pos) != 0.0,
                _ => return Err(SynaptixError::Unsupported("where: cond dtype")),
            };
            let pos = i * elem;
            let src = if is_true { &a_bytes[pos..pos + elem] } else { &b_bytes[pos..pos + elem] };
            out_bytes[pos..pos + elem].copy_from_slice(src);
        }
        let storage = Storage::Cpu(CpuBuf::from_vec(out_bytes));
        let layout = Layout::contiguous(shape, aa.dtype());
        let mut out = Tensor::from_parts(Arc::new(storage), layout);
        crate::grad::try_attach_grad_fn(
            crate::grad::GradOp::WhereCond { cond, a, b },
            &mut out,
        )?;
        Ok(out)
    }
}

fn require_cpu(t: &Tensor, op: &'static str) -> Result<()> {
    if t.device() == Device::Cpu {
        Ok(())
    } else {
        let _ = op;
        Err(SynaptixError::Unsupported("indexing op available only on CPU"))
    }
}

fn cpu_bytes(t: &Tensor) -> Result<&[u8]> {
    match t.storage() {
        Storage::Cpu(b) => {
            let off = t.layout().byte_offset();
            Ok(&b.as_bytes()[off..])
        }
        _ => Err(SynaptixError::Unsupported("indexing: non-cpu storage")),
    }
}

fn elem_size(dtype: DType) -> Result<usize> {
    if dtype.is_sub_byte() {
        return Err(SynaptixError::Unsupported("indexing: sub-byte dtype"));
    }
    Ok((dtype.size_in_bits() / 8).max(1))
}

fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_i32(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_i64(buf: &[u8], off: usize) -> i64 {
    i64::from_le_bytes([
        buf[off], buf[off + 1], buf[off + 2], buf[off + 3], buf[off + 4], buf[off + 5],
        buf[off + 6], buf[off + 7],
    ])
}

fn read_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_f64(buf: &[u8], off: usize) -> f64 {
    f64::from_le_bytes([
        buf[off], buf[off + 1], buf[off + 2], buf[off + 3], buf[off + 4], buf[off + 5],
        buf[off + 6], buf[off + 7],
    ])
}

fn add_in_place(dst: &mut [u8], src: &[u8], dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 => {
            let d = read_f32(dst, 0) + read_f32(src, 0);
            dst.copy_from_slice(&d.to_le_bytes());
        }
        DType::F64 => {
            let d = read_f64(dst, 0) + read_f64(src, 0);
            dst.copy_from_slice(&d.to_le_bytes());
        }
        DType::F16 => {
            let dv = half::f16::from_le_bytes([dst[0], dst[1]]).to_f32();
            let sv = half::f16::from_le_bytes([src[0], src[1]]).to_f32();
            let out = half::f16::from_f32(dv + sv);
            dst.copy_from_slice(&out.to_le_bytes());
        }
        DType::BF16 => {
            let dv = half::bf16::from_le_bytes([dst[0], dst[1]]).to_f32();
            let sv = half::bf16::from_le_bytes([src[0], src[1]]).to_f32();
            let out = half::bf16::from_f32(dv + sv);
            dst.copy_from_slice(&out.to_le_bytes());
        }
        _ => return Err(SynaptixError::Unsupported("scatter_add: dtype")),
    }
    Ok(())
}

fn scalar_bytes(v: f32, dtype: DType) -> Result<Vec<u8>> {
    match dtype {
        DType::F32 => Ok(v.to_le_bytes().to_vec()),
        DType::F64 => Ok((v as f64).to_le_bytes().to_vec()),
        DType::F16 => Ok(half::f16::from_f32(v).to_le_bytes().to_vec()),
        DType::BF16 => Ok(half::bf16::from_f32(v).to_le_bytes().to_vec()),
        _ => Err(SynaptixError::Unsupported("masked_fill: scalar dtype")),
    }
}
