use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;

#[derive(Clone, Debug, Default)]
pub struct InputStats {
    pub sum_sq: Vec<f64>,
    pub rows: u64,
}

impl InputStats {
    pub fn mean_sq(&self) -> Vec<f32> {
        let n = self.rows.max(1) as f64;
        self.sum_sq.iter().map(|v| (v / n) as f32).collect()
    }
}

static TABLE: Mutex<Option<std::sync::Arc<BTreeMap<String, Vec<f32>>>>> = Mutex::new(None);

pub fn set_table(t: Option<BTreeMap<String, Vec<f32>>>) {
    *TABLE.lock().unwrap() = t.map(std::sync::Arc::new);
}

pub fn table_get(weight_name: &str) -> Option<Vec<f32>> {
    TABLE.lock().unwrap().as_ref().and_then(|t| t.get(weight_name).cloned())
}

static ACTIVE: AtomicBool = AtomicBool::new(false);
static STATS: Mutex<BTreeMap<String, InputStats>> = Mutex::new(BTreeMap::new());

pub fn start() {
    STATS.lock().unwrap().clear();
    GRAMS.lock().unwrap().clear();
    ACTIVE.store(true, Ordering::Release);
}

pub fn active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

pub fn finish() -> BTreeMap<String, InputStats> {
    ACTIVE.store(false, Ordering::Release);
    std::mem::take(&mut *STATS.lock().unwrap())
}

pub fn record(weight_name: &str, x: &Tensor) -> Result<()> {
    if !active() {
        return Ok(());
    }
    let dims = x.dims().to_vec();
    let k = *dims.last().ok_or(SynaptixError::Unsupported("calib::record: скаляр"))?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let sq = x.to_dtype(DType::F32)?.reshape((rows, k))?.sqr()?.sum_keepdim(0)?;
    let v = sq.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut map = STATS.lock().unwrap();
    let e = map.entry(weight_name.to_string()).or_insert_with(|| InputStats { sum_sq: vec![0.0; k], rows: 0 });
    if e.sum_sq.len() != k {
        return Err(SynaptixError::Other(format!("calib: {weight_name}: K {k} вместо {}", e.sum_sq.len())));
    }
    for (a, b) in e.sum_sq.iter_mut().zip(&v) {
        *a += *b as f64;
    }
    e.rows += rows as u64;
    Ok(())
}

static GRAMS: Mutex<BTreeMap<String, InputStats>> = Mutex::new(BTreeMap::new());

/// Суффикс ключа матрицы вторых моментов входа (`E[x·xᵀ]`, `[k·k]` построчно).
pub const GRAM_SUFFIX: &str = "@gram";

/// Накопить `XᵀX` входа `x` `[.., k]` под именем `weight_name` — для
/// компенсации ошибки квантования (`quant::gptq`). Сумма копится на хосте в f64.
pub fn record_gram(weight_name: &str, x: &Tensor) -> Result<()> {
    if !active() {
        return Ok(());
    }
    let dims = x.dims().to_vec();
    let k = *dims.last().ok_or(SynaptixError::Unsupported("calib::record_gram: скаляр"))?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let xf = x.to_dtype(DType::F32)?.reshape((rows, k))?.contiguous()?;
    let g = xf.t()?.contiguous()?.matmul(&xf)?;
    let v = g.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut map = GRAMS.lock().unwrap();
    let e = map.entry(weight_name.to_string()).or_insert_with(|| InputStats { sum_sq: vec![0.0; k * k], rows: 0 });
    if e.sum_sq.len() != k * k {
        return Err(SynaptixError::Other(format!("calib: {weight_name}: грам {k}² вместо {}", e.sum_sq.len())));
    }
    for (a, b) in e.sum_sq.iter_mut().zip(&v) {
        *a += *b as f64;
    }
    e.rows += rows as u64;
    Ok(())
}

/// Забрать накопленные матрицы вторых моментов: ключи `<имя>@gram`.
pub fn take_grams() -> BTreeMap<String, InputStats> {
    std::mem::take(&mut *GRAMS.lock().unwrap()).into_iter().map(|(k, v)| (format!("{k}{GRAM_SUFFIX}"), v)).collect()
}

pub fn save(path: &Path, stats: &BTreeMap<String, InputStats>) -> std::io::Result<()> {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, s) in stats {
        let v = s.mean_sq();
        let start = data.len();
        for x in &v {
            data.extend_from_slice(&x.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({ "dtype": "F32", "shape": [v.len()], "data_offsets": [start, data.len()] }),
        );
    }
    let mut meta = serde_json::Map::new();
    meta.insert("format".into(), serde_json::Value::String("synaptix-calib-v1".into()));
    header.insert("__metadata__".into(), serde_json::Value::Object(meta));
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(h.len() as u64).to_le_bytes())?;
    f.write_all(&h)?;
    f.write_all(&data)?;
    Ok(())
}

pub fn load(path: &Path) -> std::io::Result<BTreeMap<String, Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < 8 {
        return Err(bad("калибровка: короткий файл"));
    }
    let hl = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + hl).ok_or_else(|| bad("заголовок"))?)?;
    let body = &bytes[8 + hl..];
    let mut out = BTreeMap::new();
    for (name, v) in header.as_object().ok_or_else(|| bad("заголовок не объект"))? {
        if name == "__metadata__" {
            continue;
        }
        let off = v["data_offsets"].as_array().ok_or_else(|| bad("data_offsets"))?;
        let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
        let raw = body.get(a..b).ok_or_else(|| bad("смещения"))?;
        out.insert(name.clone(), raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }
    Ok(out)
}
