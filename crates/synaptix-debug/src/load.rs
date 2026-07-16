use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::dump::{tag_to_dtype, TensorDump, MAGIC, VERSION};
use crate::error::{DebugError, Result};

pub fn load_from_file(path: impl AsRef<Path>) -> Result<TensorDump> {
    let p = path.as_ref();
    let file = File::open(p).map_err(|e| DebugError::Io { path: p.to_path_buf(), source: e })?;
    let mut r = BufReader::new(file);
    load_from_reader(&mut r)
}

pub fn load_from_reader(r: &mut impl Read) -> Result<TensorDump> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(DebugError::InvalidMagic { expected: MAGIC, got: magic });
    }
    let version = read_u32(r)?;
    if version != VERSION {
        return Err(DebugError::UnsupportedVersion(version));
    }
    let dtype_tag = read_u32(r)?;
    let dtype = tag_to_dtype(dtype_tag)?;
    let rank = read_u32(r)? as usize;
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(read_u64(r)? as usize);
    }
    let name_len = read_u32(r)? as usize;
    let mut name_bytes = vec![0u8; name_len];
    r.read_exact(&mut name_bytes)?;
    let name = String::from_utf8(name_bytes)
        .map_err(|e| DebugError::Other(format!("invalid utf-8 in tensor name: {e}")))?;

    let numel: usize = dims.iter().product();
    let body_len = dtype.bytes_for_numel(numel);
    let mut data = vec![0u8; body_len];
    r.read_exact(&mut data)?;

    Ok(TensorDump { name, dtype, dims, data })
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
