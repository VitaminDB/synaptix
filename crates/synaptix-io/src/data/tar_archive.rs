use std::io::Read;
use std::path::Path;

use crate::error::{IoError, Result};

pub struct TarEntry {
    pub path: String,
    pub data: Vec<u8>,
}

pub fn read_tar(path: impl AsRef<Path>) -> Result<Vec<TarEntry>> {
    let path = path.as_ref();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "gz" || path.to_string_lossy().ends_with(".tar.gz") {
        return Err(IoError::Data("gzip not supported; use .tar".into()));
    }

    let mut file = std::fs::File::open(path).map_err(IoError::Io)?;
    let mut entries = Vec::new();

    loop {
        let mut header = [0u8; 512];
        let n = read_exact_or_zero(&mut file, &mut header)?;
        if n == 0 {
            break;
        }
        if header.iter().all(|&b| b == 0) {
            let mut second = [0u8; 512];
            let _ = read_exact_or_zero(&mut file, &mut second)?;
            break;
        }

        let name = parse_tar_name(&header);
        let size = parse_tar_size(&header)?;

        if size == 0 {
            continue;
        }

        let mut data = vec![0u8; size];
        file.read_exact(&mut data).map_err(IoError::Io)?;

        let padding = (512 - (size % 512)) % 512;
        if padding > 0 {
            let mut pad_buf = vec![0u8; padding];
            file.read_exact(&mut pad_buf).map_err(IoError::Io)?;
        }

        if !name.is_empty() {
            entries.push(TarEntry { path: name, data });
        }
    }

    Ok(entries)
}

fn read_exact_or_zero(r: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        match r.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(e) => return Err(IoError::Io(e)),
        }
    }
    Ok(total)
}

fn parse_tar_name(header: &[u8; 512]) -> String {
    let raw = &header[0..100];
    let end = raw.iter().position(|&b| b == 0).unwrap_or(100);
    String::from_utf8_lossy(&raw[..end]).trim().to_string()
}

fn parse_tar_size(header: &[u8; 512]) -> Result<usize> {
    let raw = &header[124..136];
    let s = std::str::from_utf8(raw)
        .map_err(|_| IoError::Data("tar size field invalid utf8".into()))?
        .trim_matches(|c: char| c == '\0' || c.is_whitespace());
    if s.is_empty() {
        return Ok(0);
    }
    usize::from_str_radix(s, 8).map_err(|e| IoError::Data(format!("tar size octal parse: {e}")))
}
