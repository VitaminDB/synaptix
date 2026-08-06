use std::io::Write;

use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl StDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            StDtype::F64 => "F64",
            StDtype::F32 => "F32",
            StDtype::F16 => "F16",
            StDtype::BF16 => "BF16",
            StDtype::I64 => "I64",
            StDtype::I32 => "I32",
            StDtype::I16 => "I16",
            StDtype::I8 => "I8",
            StDtype::U8 => "U8",
            StDtype::Bool => "BOOL",
        }
    }

    pub fn size(self) -> usize {
        match self {
            StDtype::F64 | StDtype::I64 => 8,
            StDtype::F32 | StDtype::I32 => 4,
            StDtype::F16 | StDtype::BF16 | StDtype::I16 => 2,
            StDtype::I8 | StDtype::U8 | StDtype::Bool => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamTensor {
    pub name: String,
    pub dtype: StDtype,
    pub shape: Vec<usize>,
}

impl StreamTensor {
    pub fn nbytes(&self) -> u64 {
        self.shape.iter().product::<usize>() as u64 * self.dtype.size() as u64
    }
}

pub trait TensorStream: Send {
    fn plan(&self) -> &[StreamTensor];
    fn write_tensor(&mut self, index: usize, w: &mut dyn Write) -> Result<()>;
}

pub fn safetensors_header(plan: &[StreamTensor], align: usize) -> Result<Vec<u8>> {
    use std::fmt::Write as _;

    let mut offset: u64 = 0;
    let mut json = String::from("{");
    for (i, t) in plan.iter().enumerate() {
        let n = t.nbytes();
        if i > 0 {
            json.push(',');
        }
        let shape = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let name = serde_json::to_string(&t.name).map_err(|e| {
            crate::error::Error::Safetensors(format!("имя тензора не сериализуется: {e}"))
        })?;
        let _ = write!(
            json,
            "{name}:{{\"dtype\":\"{}\",\"shape\":[{shape}],\"data_offsets\":[{offset},{}]}}",
            t.dtype.as_str(),
            offset + n
        );
        offset += n;
    }
    json.push('}');

    let mut len = json.len();
    let align = align.max(1);

    while (8 + len) % align != 0 {
        len += 1;
    }
    let mut out = Vec::with_capacity(8 + len);
    out.extend_from_slice(&(len as u64).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    out.resize(8 + len, b' ');
    Ok(out)
}

pub fn payload_len(plan: &[StreamTensor], align: usize) -> Result<u64> {
    let hdr = safetensors_header(plan, align)?;
    let data: u64 = plan.iter().map(|t| t.nbytes()).sum();
    Ok(hdr.len() as u64 + data)
}

pub(crate) struct CountingWriter<'a, W: Write> {
    pub inner: &'a mut W,
    pub written: u64,
    pub crc: u32,
    pub on_bytes: Option<&'a mut dyn FnMut(u64)>,
}

impl<'a, W: Write> Write for CountingWriter<'a, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write_all(buf)?;
        self.written += buf.len() as u64;
        self.crc = crc32c::crc32c_append(self.crc, buf);
        if let Some(cb) = self.on_bytes.as_mut() {
            cb(buf.len() as u64);
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Vec<StreamTensor> {
        vec![
            StreamTensor {
                name: "a.weight".into(),
                dtype: StDtype::BF16,
                shape: vec![2, 3],
            },
            StreamTensor {
                name: "b.weight".into(),
                dtype: StDtype::F32,
                shape: vec![4],
            },
        ]
    }

    #[test]
    fn header_is_valid_safetensors_and_aligned() {
        let p = plan();
        let hdr = safetensors_header(&p, 64).unwrap();
        assert_eq!(hdr.len() % 64, 0);
        let n = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
        assert_eq!(8 + n, hdr.len());
        let json: serde_json::Value = serde_json::from_slice(hdr[8..].trim_ascii_end()).unwrap();
        assert_eq!(json["a.weight"]["dtype"], "BF16");
        assert_eq!(json["a.weight"]["data_offsets"][1], 12);
        assert_eq!(json["b.weight"]["data_offsets"][0], 12);
        assert_eq!(json["b.weight"]["data_offsets"][1], 28);
    }

    #[test]
    fn payload_len_matches_header_plus_data() {
        let p = plan();
        let hdr = safetensors_header(&p, 64).unwrap();
        assert_eq!(payload_len(&p, 64).unwrap(), hdr.len() as u64 + 12 + 16);
    }

    #[test]
    fn round_trips_through_safetensors_crate() {
        let p = plan();
        let mut buf = safetensors_header(&p, 64).unwrap();
        buf.extend_from_slice(&[0u8; 12]);
        buf.extend_from_slice(&[0u8; 16]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut names = st.names();
        names.sort();
        assert_eq!(names, vec!["a.weight", "b.weight"]);
        assert_eq!(st.tensor("a.weight").unwrap().shape(), &[2, 3]);
    }
}
