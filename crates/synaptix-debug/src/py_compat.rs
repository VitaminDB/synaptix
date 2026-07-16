use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_core::tensor::storage::Storage;

use crate::error::{DebugError, Result};

pub fn save_npy(tensor: &Tensor, path: impl AsRef<Path>) -> Result<()> {
    let p = path.as_ref();
    let f = File::create(p).map_err(|e| DebugError::Io { path: p.to_path_buf(), source: e })?;
    let mut w = BufWriter::new(f);
    write_npy(tensor, &mut w)
}

fn write_npy(tensor: &Tensor, w: &mut impl Write) -> Result<()> {
    let t = tensor.contiguous()?;
    let dtype_descr = npy_dtype_descr(t.dtype())?;
    let dims = t.dims();
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': (", dtype_descr);
    for (i, d) in dims.iter().enumerate() {
        if i > 0 {
            header.push_str(", ");
        }
        header.push_str(&d.to_string());
    }
    if dims.len() == 1 {
        header.push(',');
    }
    header.push_str(")}");
    let total_unpadded = 10 + header.len() + 1;
    let pad = (64 - total_unpadded % 64) % 64;
    for _ in 0..pad {
        header.push(' ');
    }
    header.push('\n');

    w.write_all(&[0x93])?;
    w.write_all(b"NUMPY")?;
    w.write_all(&[1u8, 0u8])?;
    let hlen = header.len() as u16;
    w.write_all(&hlen.to_le_bytes())?;
    w.write_all(header.as_bytes())?;

    let storage = t.storage();
    let Storage::Cpu(buf) = storage else {
        return Err(DebugError::Other("save_npy: non-cpu storage".into()));
    };
    let off = t.layout().byte_offset();
    let body_len = t.dtype().bytes_for_numel(t.numel());
    w.write_all(&buf.as_bytes()[off..off + body_len])?;
    w.flush()?;
    Ok(())
}

fn npy_dtype_descr(dt: DType) -> Result<&'static str> {
    Ok(match dt {
        DType::F32 => "<f4",
        DType::F64 => "<f8",
        DType::F16 => "<f2",
        DType::BF16 => {
            return Err(DebugError::Other(
                "save_npy: BF16 не поддерживается numpy без расширения".into(),
            ));
        }
        DType::U8 => "|u1",
        DType::U32 => "<u4",
        DType::I32 => "<i4",
        DType::I64 => "<i8",
        _ => return Err(DebugError::Other(format!("save_npy: dtype {dt:?}"))),
    })
}
