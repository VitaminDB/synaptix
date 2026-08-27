use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::pos::rope_cache::RopeCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    Interleaved,
    Split,
}

pub fn apply_rope(
    x: &Tensor,
    cache: &RopeCache,
    positions: Option<&Tensor>,
    layout: RopeLayout,
) -> Result<Tensor> {
    if x.rank() != 4 {
        return Err(SynaptixError::Unsupported("apply_rope: requires rank-4 [B,H,S,D]"));
    }
    let head_dim = x.dims()[3];
    if head_dim != cache.head_dim() {
        return Err(SynaptixError::Other(format!(
            "apply_rope: head_dim {} mismatch cache {}",
            head_dim,
            cache.head_dim()
        )));
    }
    let s = x.dims()[2];
    let (cos, sin) = match positions {
        Some(p) => cache.select_positions(p)?,
        None => cache.select_range(0, s)?,
    };
    apply_rope_with_cossin(x, &cos, &sin, layout)
}

/// RoPE для смежного диапазона позиций `[start, start+len)`. В отличие от
/// [`apply_rope`] с `Some(positions)`, выбирает cos/sin через `select_range`
/// (narrow — без `index_select`/`clone_dtoh` host-round-trip). Для prefill
/// позиции всегда смежны (`past..past+s`), поэтому это горячий путь.
pub fn apply_rope_range(
    x: &Tensor,
    cache: &RopeCache,
    start: usize,
    len: usize,
    layout: RopeLayout,
) -> Result<Tensor> {
    if x.rank() != 4 {
        return Err(SynaptixError::Unsupported("apply_rope_range: requires rank-4 [B,H,S,D]"));
    }
    let head_dim = x.dims()[3];
    if head_dim != cache.head_dim() {
        return Err(SynaptixError::Other(format!(
            "apply_rope_range: head_dim {} mismatch cache {}",
            head_dim,
            cache.head_dim()
        )));
    }
    debug_assert_eq!(x.dims()[2], len, "apply_rope_range: seq mismatch");
    let (cos, sin) = cache.select_range(start, len)?;
    apply_rope_with_cossin(x, &cos, &sin, layout)
}

/// RoPE по готовым таблицам `cos`/`sin` формы `[S, head_dim/2]` (F32) —
/// строка на позицию токена. Нужен там, где позиции не последовательны
/// (M-RoPE: у каждой частоты своя ось позиции — таблицы строятся снаружи).
pub fn apply_rope_with_cossin(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    layout: RopeLayout,
) -> Result<Tensor> {
    let head_dim = x.dims()[3];
    let s = x.dims()[2];
    // Fused backend-путь (CUDA — один launch вместо ~12 decomposed-ops). cos/sin
    // уже [S, head_dim/2] F32. На CPU/неподдержке → decomposed ниже.
    if layout == RopeLayout::Split {
        match x.rope_split_fused(cos, sin) {
            Ok(out) => return Ok(out),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
    }
    let cos_b = cos.reshape((1usize, 1, s, head_dim / 2))?;
    let sin_b = sin.reshape((1usize, 1, s, head_dim / 2))?;
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let half = head_dim / 2;
    let (x_a, x_b) = match layout {
        RopeLayout::Split => (
            x_f32.narrow(3, 0, half)?.contiguous()?,
            x_f32.narrow(3, half, half)?.contiguous()?,
        ),
        RopeLayout::Interleaved => split_interleaved(&x_f32, half)?,
    };
    let rot_a = x_a.broadcast_mul(&cos_b)?.sub(&x_b.broadcast_mul(&sin_b)?)?;
    let rot_b = x_a.broadcast_mul(&sin_b)?.add(&x_b.broadcast_mul(&cos_b)?)?;
    let out = match layout {
        RopeLayout::Split => Tensor::cat(&[&rot_a, &rot_b], 3)?,
        RopeLayout::Interleaved => interleave(&rot_a, &rot_b)?,
    };
    out.to_dtype(dtype_in)
}

fn split_interleaved(x: &Tensor, half: usize) -> Result<(Tensor, Tensor)> {
    let b = x.dims()[0];
    let h = x.dims()[1];
    let s = x.dims()[2];
    let reshaped = x.reshape((b, h, s, half, 2))?;
    let a = reshaped.narrow(4, 0, 1)?.contiguous()?.reshape((b, h, s, half))?;
    let bvec = reshaped.narrow(4, 1, 1)?.contiguous()?.reshape((b, h, s, half))?;
    Ok((a, bvec))
}

fn interleave(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let bn = a.dims()[0];
    let hn = a.dims()[1];
    let sn = a.dims()[2];
    let half = a.dims()[3];
    let a_unsq = a.reshape((bn, hn, sn, half, 1))?;
    let b_unsq = b.reshape((bn, hn, sn, half, 1))?;
    let stacked = Tensor::cat(&[&a_unsq, &b_unsq], 4)?;
    stacked.reshape((bn, hn, sn, half * 2))
}

pub fn apply_rope_split(
    x: &Tensor,
    cache: &RopeCache,
    positions: Option<&Tensor>,
) -> Result<Tensor> {
    apply_rope(x, cache, positions, RopeLayout::Split)
}

pub fn apply_rope_interleaved(
    x: &Tensor,
    cache: &RopeCache,
    positions: Option<&Tensor>,
) -> Result<Tensor> {
    apply_rope(x, cache, positions, RopeLayout::Interleaved)
}
