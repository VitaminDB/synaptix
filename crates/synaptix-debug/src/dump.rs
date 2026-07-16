use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_core::tensor::storage::Storage;

use crate::error::{DebugError, Result};

pub const MAGIC: [u8; 8] = *b"SYNDMP01";
pub const VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct TensorDump {
    pub name: String,
    pub dtype: DType,
    pub dims: Vec<usize>,
    pub data: Vec<u8>,
}

impl TensorDump {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

pub fn dump_to_file(tensor: &Tensor, name: &str, path: impl AsRef<Path>) -> Result<()> {
    let p = path.as_ref();
    let file = File::create(p).map_err(|e| DebugError::Io { path: p.to_path_buf(), source: e })?;
    let mut w = BufWriter::new(file);
    dump_to_writer(tensor, name, &mut w)
}

pub fn dump_to_writer(tensor: &Tensor, name: &str, w: &mut impl Write) -> Result<()> {
    let contig = tensor.contiguous()?;
    let dtype = contig.dtype();
    let dims = contig.dims().to_vec();
    let raw_bytes = cpu_bytes(&contig)?;

    w.write_all(&MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&dtype_tag(dtype).to_le_bytes())?;
    w.write_all(&(dims.len() as u32).to_le_bytes())?;
    for &d in &dims {
        w.write_all(&(d as u64).to_le_bytes())?;
    }
    let name_bytes = name.as_bytes();
    w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
    w.write_all(name_bytes)?;

    let expected_len = dtype.bytes_for_numel(dims.iter().product());
    if raw_bytes.len() < expected_len {
        return Err(DebugError::Other(format!(
            "internal: cpu_bytes returned {} bytes, expected at least {}",
            raw_bytes.len(),
            expected_len
        )));
    }
    w.write_all(&raw_bytes[..expected_len])?;
    w.flush()?;
    Ok(())
}

fn cpu_bytes(t: &Tensor) -> Result<Vec<u8>> {
    let storage = t.storage();
    match storage {
        Storage::Cpu(buf) => {
            let off = t.layout().byte_offset();
            Ok(buf.as_bytes()[off..].to_vec())
        }
        _ => Err(DebugError::Other("non-CPU storage not supported in TensorDump".into())),
    }
}

pub(crate) fn dtype_tag(dt: DType) -> u32 {
    match dt {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::U8 => 10,
        DType::U32 => 11,
        DType::I32 => 12,
        DType::I64 => 13,
        DType::NVFP4 => 21,
        DType::MXFP8 => 22,
    }
}

pub(crate) fn tag_to_dtype(tag: u32) -> Result<DType> {
    match tag {
        0 => Ok(DType::F32),
        1 => Ok(DType::F64),
        2 => Ok(DType::F16),
        3 => Ok(DType::BF16),
        10 => Ok(DType::U8),
        11 => Ok(DType::U32),
        12 => Ok(DType::I32),
        13 => Ok(DType::I64),
        21 => Ok(DType::NVFP4),
        22 => Ok(DType::MXFP8),
        _ => Err(DebugError::UnknownDtype(tag)),
    }
}
