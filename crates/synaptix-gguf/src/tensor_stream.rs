use std::io::Write;
use std::sync::Arc;

use half::{bf16, f16};
use rayon::prelude::*;
use synaptix_bundle::inspect::QuantKind;
use synaptix_bundle::{QuantEntry, QuantManifest, StDtype, StreamTensor, TensorStream};

use crate::dequant::dequantize;
use crate::error::{GgufError, Result};
use crate::ggml::GgmlType;
use crate::plan::{Component, MappedTensor, OutDtype, Producer, Transform};
use crate::reader::GgufFile;

const WINDOW_ELEMS: usize = 4 << 20;

struct Item {
    producer: Producer,
    transform: Transform,

    src_ty: GgmlType,
    out: StDtype,
    /// `OutDtype::Keep`: тензор пишется блоками ggml как есть —
    /// `(число матриц, N, K)`; блоб идёт в `<имя>.qpacked`.
    keep: Option<(usize, usize, usize)>,
}

pub struct GgufTensorStream {
    files: Vec<Arc<GgufFile>>,
    plan: Vec<StreamTensor>,
    items: Vec<Item>,
    /// Манифест квантованных тензоров (`quant_manifest.json`) — заполняется
    /// только в режиме `Keep`.
    manifest: QuantManifest,
}

/// Форма тензора в HF-порядке: из плана или из первого источника (стопка из
/// частей складывает `N`).
pub(crate) fn hf_shape_of(files: &[Arc<GgufFile>], m: &MappedTensor) -> Option<Vec<usize>> {
    if let Some(s) = &m.shape {
        return Some(s.clone());
    }
    match &m.producer {
        Producer::StackConcat { parts } => {
            let shapes: Vec<Vec<usize>> = parts.iter().filter_map(|p| lookup(files, p).ok().map(|i| i.hf_shape())).collect();
            if shapes.len() != parts.len() || shapes.iter().any(|s| s.len() != 3) {
                return None;
            }
            let e = shapes[0][0];
            let k = shapes[0][2];
            let n: usize = shapes.iter().map(|s| s[1]).sum();
            Some(vec![e, n, k])
        }
        _ => lookup(files, &m.producer.sources()[0]).ok().map(|i| i.hf_shape()),
    }
}

/// Можно ли отдать тензор блоками ggml без деквантования: квантованный
/// формат весов, матрица или стопка, K кратен блоку и 32, продюсер не
/// режет строки внутри блока, преобразований нет. Возвращает
/// `(тип, число матриц, N, K)`.
pub(crate) fn quant_capable(files: &[Arc<GgufFile>], m: &MappedTensor, ty: GgmlType) -> Option<(GgmlType, usize, usize, usize)> {
    if m.transform != Transform::None {
        return None;
    }
    if !ty.is_weight_format() {
        return None;
    }
    let shape = hf_shape_of(files, m)?;
    let (slices, n, k) = match shape.as_slice() {
        [n, k] => (1usize, *n, *k),
        [e, n, k] => (*e, *n, *k),
        _ => return None,
    };
    let be = ty.block_elems();
    if k % be != 0 || k % 32 != 0 || n == 0 || slices == 0 {
        return None;
    }
    match &m.producer {
        Producer::Direct(_) | Producer::StackConcat { .. } => {}
        Producer::PermuteRows { row_elems, .. } => {
            if *row_elems != k {
                return None;
            }
        }
        Producer::PermuteCols { block, .. } => {
            if block % be != 0 {
                return None;
            }
        }
        Producer::Interleave { .. } => return None,
    }
    Some((ty, slices, n, k))
}

/// Байты среза `slice` квантованного веса `[N, K]`: заимствованные (Direct)
/// или собранные на уровне блоков (перестановки строк/столбцов, стопка из
/// частей).
pub(crate) fn quant_blob<'a>(
    files: &'a [Arc<GgufFile>],
    m: &MappedTensor,
    ty: GgmlType,
    slice: usize,
    n: usize,
    k: usize,
) -> Result<std::borrow::Cow<'a, [u8]>> {
    use std::borrow::Cow;
    let rb = ty.bytes_for(k);
    let slice_bytes = n * rb;
    match &m.producer {
        Producer::Direct(src) => {
            let all = bytes_of(files, src)?;
            let off = slice * slice_bytes;
            if off + slice_bytes > all.len() {
                return Err(GgufError::BadTensor { name: m.hf_name.clone(), reason: "срез за границей блоба".into() });
            }
            Ok(Cow::Borrowed(&all[off..off + slice_bytes]))
        }
        Producer::PermuteRows { src, map, .. } => {
            let all = bytes_of(files, src)?;
            let mut out = Vec::with_capacity(map.len() * rb);
            for r in map {
                let o = *r as usize * rb;
                out.extend_from_slice(&all[o..o + rb]);
            }
            Ok(Cow::Owned(out))
        }
        Producer::PermuteCols { src, row_elems, block, map } => {
            let all = bytes_of(files, src)?;
            let be = ty.block_elems();
            let bb = ty.block_bytes();
            let src_rb = row_elems / be * bb;
            let blk_bytes = block / be * bb;
            let rows = all.len() / src_rb;
            let mut out = Vec::with_capacity(rows * map.len() * blk_bytes);
            for r in 0..rows {
                let base = r * src_rb;
                for c in map {
                    let o = base + *c as usize * blk_bytes;
                    out.extend_from_slice(&all[o..o + blk_bytes]);
                }
            }
            Ok(Cow::Owned(out))
        }
        Producer::StackConcat { parts } => {
            let mut out = Vec::with_capacity(slice_bytes);
            for p in parts {
                let info = lookup(files, p)?;
                let shape = info.hf_shape();
                let per = shape[1] * rb;
                let all = bytes_of(files, p)?;
                out.extend_from_slice(&all[slice * per..(slice + 1) * per]);
            }
            Ok(Cow::Owned(out))
        }
        Producer::Interleave { .. } => Err(GgufError::BadTensor { name: m.hf_name.clone(), reason: "interleave не отдаётся блоками".into() }),
    }
}

impl GgufTensorStream {

    pub fn new(files: Vec<Arc<GgufFile>>, comp: &Component, dtype: OutDtype) -> Result<Self> {
        let mut plan = Vec::with_capacity(comp.tensors.len());
        let mut items = Vec::with_capacity(comp.tensors.len());
        let mut manifest = QuantManifest::new();
        for mt in &comp.tensors {
            let srcs = mt.producer.sources();
            let first = lookup(&files, &srcs[0])?;
            let src_ty = first.ty;
            let mut elems = 0usize;
            for s in srcs {
                let info = lookup(&files, s)?;
                if info.ty != src_ty {
                    return Err(GgufError::BadTensor {
                        name: mt.hf_name.clone(),
                        reason: format!(
                            "части имеют разные ggml-типы: {} и {}",
                            src_ty.name(),
                            info.ty.name()
                        ),
                    });
                }
                elems += info.elem_count();
            }
            let shape = match &mt.shape {
                Some(s) => s.clone(),
                None => first.hf_shape(),
            };
            let shape_elems: usize = shape.iter().product();
            if shape_elems != elems {
                return Err(GgufError::BadTensor {
                    name: mt.hf_name.clone(),
                    reason: format!("форма {shape:?} даёт {shape_elems} элементов, источники — {elems}"),
                });
            }
            // Keep: квантованный вес остаётся блоками ggml — блоб
            // `<имя>.qpacked` (U8, `[.., N, row_bytes]`) и запись в манифесте
            // с исходной формой. Всё, что блоками не отдаётся (нормы,
            // interleave, преобразования), деквантуется как в Auto.
            let keep = if dtype.keeps_quant(src_ty) { quant_capable(&files, mt, src_ty) } else { None };
            if let Some((ty, slices, n, k)) = keep {
                let rb = ty.bytes_for(k);
                let blob_shape = if slices == 1 { vec![n, rb] } else { vec![slices, n, rb] };
                manifest.tensors.insert(
                    mt.hf_name.clone(),
                    QuantEntry { format: synaptix_bundle::quant_layout::format_key(QuantKind::Ggml(ty)), shape: shape.clone() },
                );
                plan.push(StreamTensor { name: manifest.packed_name(&mt.hf_name), dtype: StDtype::U8, shape: blob_shape });
                items.push(Item {
                    producer: mt.producer.clone(),
                    transform: mt.transform,
                    src_ty,
                    out: StDtype::U8,
                    keep: Some((slices, n, k)),
                });
                continue;
            }
            plan.push(StreamTensor {
                name: mt.hf_name.clone(),
                dtype: dtype.resolve(src_ty),
                shape,
            });
            items.push(Item {
                producer: mt.producer.clone(),
                transform: mt.transform,
                src_ty,
                out: dtype.resolve(src_ty),
                keep: None,
            });
        }
        Ok(Self { files, plan, items, manifest })
    }

    /// Квантованные тензоры, оставленные блоками (`OutDtype::Keep`); пусто в
    /// остальных режимах.
    pub fn quant_manifest(&self) -> &QuantManifest {
        &self.manifest
    }
}

pub(crate) fn lookup<'a>(files: &'a [Arc<GgufFile>], name: &str) -> Result<&'a crate::reader::TensorInfo> {
    for f in files {
        if let Some(t) = f.tensor(name) {
            return Ok(t);
        }
    }
    Err(GgufError::BadTensor {
        name: name.to_string(),
        reason: "нет в GGUF-источниках".into(),
    })
}

pub(crate) fn bytes_of<'a>(files: &'a [Arc<GgufFile>], name: &str) -> Result<&'a [u8]> {
    for f in files {
        if let Some(t) = f.tensor(name) {
            return f.tensor_bytes(t);
        }
    }
    Err(GgufError::BadTensor {
        name: name.to_string(),
        reason: "нет в GGUF-источниках".into(),
    })
}

#[inline]
pub(crate) fn apply_transform(t: Transform, v: &mut [f32]) {
    match t {
        Transform::None => {}
        Transform::LogNeg => {
            for x in v.iter_mut() {
                *x = (-*x).ln();
            }
        }
        Transform::SubOne => {
            for x in v.iter_mut() {
                *x -= 1.0;
            }
        }
    }
}

fn encode(out: StDtype, src: &[f32], dst: &mut Vec<u8>) {
    dst.clear();
    match out {
        StDtype::F32 => {
            dst.reserve(src.len() * 4);
            for v in src {
                dst.extend_from_slice(&v.to_le_bytes());
            }
        }
        StDtype::F16 => {
            dst.reserve(src.len() * 2);
            for v in src {
                dst.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
            }
        }
        StDtype::BF16 => {
            dst.reserve(src.len() * 2);
            for v in src {
                dst.extend_from_slice(&bf16::from_f32(*v).to_le_bytes());
            }
        }
        StDtype::F64 => {
            dst.reserve(src.len() * 8);
            for v in src {
                dst.extend_from_slice(&(*v as f64).to_le_bytes());
            }
        }
        StDtype::I64 => {
            for v in src {
                dst.extend_from_slice(&(*v as i64).to_le_bytes());
            }
        }
        StDtype::I32 => {
            for v in src {
                dst.extend_from_slice(&(*v as i32).to_le_bytes());
            }
        }
        StDtype::I16 => {
            for v in src {
                dst.extend_from_slice(&(*v as i16).to_le_bytes());
            }
        }
        StDtype::I8 | StDtype::U8 | StDtype::Bool => {
            for v in src {
                dst.push(*v as i8 as u8);
            }
        }
    }
}

pub(crate) fn dequant_parallel(ty: GgmlType, src: &[u8], n: usize, dst: &mut [f32]) -> Result<()> {
    let be = ty.block_elems();
    let bb = ty.block_bytes();

    let blocks = n.div_ceil(be);
    let per_task = (blocks / rayon::current_num_threads().max(1)).max(64);
    let elems_per_task = per_task * be;
    let bytes_per_task = per_task * bb;

    let results: Vec<Result<()>> = dst[..n]
        .par_chunks_mut(elems_per_task)
        .zip(src.par_chunks(bytes_per_task))
        .map(|(d, s)| {
            let cnt = d.len();
            dequantize(ty, s, cnt, d)
        })
        .collect();
    for r in results {
        r?;
    }
    Ok(())
}

impl TensorStream for GgufTensorStream {
    fn plan(&self) -> &[StreamTensor] {
        &self.plan
    }

    fn write_tensor(
        &mut self,
        index: usize,
        w: &mut dyn Write,
    ) -> std::result::Result<(), synaptix_bundle::Error> {
        self.write_one(index, w)
            .map_err(|e| synaptix_bundle::Error::Safetensors(e.to_string()))
    }
}

impl GgufTensorStream {
    fn write_one(&mut self, index: usize, w: &mut dyn Write) -> Result<()> {
        let item = &self.items[index];
        if let Some((slices, n, k)) = item.keep {
            // Блоки как есть, срез за срезом (стопка экспертов — подряд).
            let m = MappedTensor {
                hf_name: self.plan[index].name.clone(),
                producer: item.producer.clone(),
                shape: None,
                transform: Transform::None,
            };
            for slice in 0..slices {
                let blob = quant_blob(&self.files, &m, item.src_ty, slice, n, k)?;
                w.write_all(&blob)?;
            }
            return Ok(());
        }
        match &item.producer {
            Producer::Direct(name) => {
                let info = lookup(&self.files, name)?;
                let n = info.elem_count();
                let src = bytes_of(&self.files, name)?;
                self.stream_span(item, src, n, w)
            }
            Producer::PermuteRows { src, row_elems, map } => {
                let buf = self.materialise(item, src)?;
                let mut out = Vec::new();
                for r in map {
                    let off = *r as usize * row_elems;
                    encode(item.out, &buf[off..off + row_elems], &mut out);
                    w.write_all(&out)?;
                }
                Ok(())
            }
            Producer::PermuteCols {
                src,
                row_elems,
                block,
                map,
            } => {
                let buf = self.materialise(item, src)?;
                let rows = buf.len() / row_elems;
                let mut out = Vec::new();
                let mut row = vec![0f32; *row_elems];
                for r in 0..rows {
                    let base = r * row_elems;
                    for (j, c) in map.iter().enumerate() {
                        let src_off = base + *c as usize * block;
                        row[j * block..(j + 1) * block]
                            .copy_from_slice(&buf[src_off..src_off + block]);
                    }
                    encode(item.out, &row, &mut out);
                    w.write_all(&out)?;
                }
                Ok(())
            }
            Producer::StackConcat { parts } => {
                // Срез e каждой части подряд: [E, N_i, K] → строки e·N_i..(e+1)·N_i.
                let mut infos = Vec::with_capacity(parts.len());
                for p in parts {
                    infos.push((p.clone(), lookup(&self.files, p)?.hf_shape()));
                }
                let experts = infos[0].1[0];
                for e in 0..experts {
                    for (p, shape) in &infos {
                        let per = shape[1..].iter().product::<usize>();
                        let src = bytes_of(&self.files, p)?;
                        let bb = item.src_ty.block_bytes();
                        let be = item.src_ty.block_elems();
                        let byte_off = per / be * bb * e;
                        let byte_len = per / be * bb;
                        self.stream_span(item, &src[byte_off..byte_off + byte_len], per, w)?;
                    }
                }
                Ok(())
            }
            Producer::Interleave { parts, block } => {

                let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(parts.len());
                for p in parts {
                    let info = lookup(&self.files, p)?;
                    let n = info.elem_count();
                    let mut v = vec![0f32; n];
                    dequant_parallel(item.src_ty, bytes_of(&self.files, p)?, n, &mut v)?;
                    apply_transform(item.transform, &mut v);
                    bufs.push(v);
                }
                let groups = bufs[0].len() / block;
                let mut out = Vec::new();
                let mut chunk = Vec::with_capacity(*block);
                for g in 0..groups {
                    for b in &bufs {
                        chunk.clear();
                        chunk.extend_from_slice(&b[g * block..(g + 1) * block]);
                        encode(item.out, &chunk, &mut out);
                        w.write_all(&out)?;
                    }
                }
                Ok(())
            }
        }
    }

    fn materialise(&self, item: &Item, name: &str) -> Result<Vec<f32>> {
        let info = lookup(&self.files, name)?;
        let n = info.elem_count();
        let mut buf = vec![0f32; n];
        dequant_parallel(item.src_ty, bytes_of(&self.files, name)?, n, &mut buf)?;
        apply_transform(item.transform, &mut buf);
        Ok(buf)
    }

    fn stream_span(
        &self,
        item: &Item,
        src: &[u8],
        n: usize,
        w: &mut dyn Write,
    ) -> Result<()> {
        let be = item.src_ty.block_elems();
        let bb = item.src_ty.block_bytes();
        let window = (WINDOW_ELEMS / be).max(1) * be;
        let mut f32buf = vec![0f32; window.min(n)];
        let mut bytes = Vec::new();
        let mut done = 0usize;
        while done < n {
            let take = window.min(n - done);
            let src_off = done / be * bb;
            let src_len = take.div_ceil(be) * bb;
            let chunk = &src[src_off..src_off + src_len];
            dequant_parallel(item.src_ty, chunk, take, &mut f32buf[..take])?;
            apply_transform(item.transform, &mut f32buf[..take]);
            encode(item.out, &f32buf[..take], &mut bytes);
            w.write_all(&bytes)?;
            done += take;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_f16_is_exact_for_small_ints() {
        let mut out = Vec::new();
        encode(StDtype::F16, &[1.0, -2.0, 300.0], &mut out);
        assert_eq!(out.len(), 6);
        assert_eq!(f16::from_le_bytes([out[4], out[5]]).to_f32(), 300.0);
    }

    #[test]
    fn log_neg_inverts_neg_exp() {
        let mut v = vec![-(2.0f32.exp()), -(0.5f32.exp())];
        apply_transform(Transform::LogNeg, &mut v);
        assert!((v[0] - 2.0).abs() < 1e-6);
        assert!((v[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dequant_parallel_matches_serial() {
        let ty = GgmlType::Q8_0;
        let nblocks = 300;
        let n = nblocks * 32;
        let mut src = vec![0u8; nblocks * 34];
        for b in 0..nblocks {
            let d = f16::from_f32(0.5 + b as f32 * 0.001);
            src[b * 34..b * 34 + 2].copy_from_slice(&d.to_le_bytes());
            for j in 0..32 {
                src[b * 34 + 2 + j] = ((b + j) as i32 % 127) as i8 as u8;
            }
        }
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        dequantize(ty, &src, n, &mut a).unwrap();
        dequant_parallel(ty, &src, n, &mut b).unwrap();
        assert_eq!(a, b);
    }
}

/// Плотная материализация тензора плана в f32 (деквант + преобразование +
/// перестановки) — для рантайм-источника GGUF, когда вес нужен плотным.
pub(crate) fn materialize_f32(
    files: &[Arc<GgufFile>],
    producer: &Producer,
    transform: Transform,
    src_ty: GgmlType,
) -> Result<Vec<f32>> {
    let mat = |name: &str| -> Result<Vec<f32>> {
        let info = lookup(files, name)?;
        let n = info.elem_count();
        let mut buf = vec![0f32; n];
        dequant_parallel(src_ty, bytes_of(files, name)?, n, &mut buf)?;
        apply_transform(transform, &mut buf);
        Ok(buf)
    };
    match producer {
        Producer::Direct(name) => mat(name),
        Producer::PermuteRows { src, row_elems, map } => {
            let buf = mat(src)?;
            let mut out = Vec::with_capacity(map.len() * row_elems);
            for r in map {
                let off = *r as usize * row_elems;
                out.extend_from_slice(&buf[off..off + row_elems]);
            }
            Ok(out)
        }
        Producer::PermuteCols { src, row_elems, block, map } => {
            let buf = mat(src)?;
            let rows = buf.len() / row_elems;
            let mut out = Vec::with_capacity(buf.len());
            for r in 0..rows {
                let base = r * row_elems;
                for c in map {
                    let so = base + *c as usize * block;
                    out.extend_from_slice(&buf[so..so + block]);
                }
            }
            Ok(out)
        }
        Producer::Interleave { parts, block } => {
            let bufs: Vec<Vec<f32>> = parts.iter().map(|p| mat(p)).collect::<Result<_>>()?;
            let groups = bufs[0].len() / block;
            let mut out = Vec::with_capacity(bufs.iter().map(|b| b.len()).sum());
            for g in 0..groups {
                for b in &bufs {
                    out.extend_from_slice(&b[g * block..(g + 1) * block]);
                }
            }
            Ok(out)
        }
        Producer::StackConcat { parts } => {
            let mut infos = Vec::with_capacity(parts.len());
            for p in parts {
                infos.push((mat(p)?, lookup(files, p)?.hf_shape()));
            }
            let experts = infos[0].1[0];
            let mut out = Vec::new();
            for e in 0..experts {
                for (buf, shape) in &infos {
                    let per = shape[1..].iter().product::<usize>();
                    out.extend_from_slice(&buf[e * per..(e + 1) * per]);
                }
            }
            Ok(out)
        }
    }
}
