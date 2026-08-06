use std::io::Write;
use std::sync::Arc;

use half::{bf16, f16};
use rayon::prelude::*;
use synaptix_bundle::{StDtype, StreamTensor, TensorStream};

use crate::dequant::dequantize;
use crate::error::{GgufError, Result};
use crate::ggml::GgmlType;
use crate::plan::{Component, OutDtype, Producer, Transform};
use crate::reader::GgufFile;

const WINDOW_ELEMS: usize = 4 << 20;

struct Item {
    producer: Producer,
    transform: Transform,

    src_ty: GgmlType,
    out: StDtype,
}

pub struct GgufTensorStream {
    files: Vec<Arc<GgufFile>>,
    plan: Vec<StreamTensor>,
    items: Vec<Item>,
}

impl GgufTensorStream {

    pub fn new(files: Vec<Arc<GgufFile>>, comp: &Component, dtype: OutDtype) -> Result<Self> {
        let mut plan = Vec::with_capacity(comp.tensors.len());
        let mut items = Vec::with_capacity(comp.tensors.len());
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
            });
        }
        Ok(Self { files, plan, items })
    }
}

fn lookup<'a>(files: &'a [Arc<GgufFile>], name: &str) -> Result<&'a crate::reader::TensorInfo> {
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

fn bytes_of<'a>(files: &'a [Arc<GgufFile>], name: &str) -> Result<&'a [u8]> {
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
fn apply_transform(t: Transform, v: &mut [f32]) {
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

fn dequant_parallel(ty: GgmlType, src: &[u8], n: usize, dst: &mut [f32]) -> Result<()> {
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
