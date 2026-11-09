//! Внешний источник исходных весов для перекодировки бандла: каталог HF
//! (`model.safetensors.index.json` + шарды). Нужен, когда в бандле веса уже
//! квантованы, а перекодировать их надо из оригинала, без двойного кванта.
//!
//! Понимает:
//! - плотные BF16/F16/F32;
//! - FP8 E4M3 с блочными масштабами `<имя>_scale_inv` (обычно 128×128,
//!   `w = fp8 · scale_inv[r/bn, c/bk]`, как в чекпойнтах DeepSeek/Qwen-FP8);
//! - стопки экспертов бандла (`….experts.gate_up_proj` `[E, 2I, H]`,
//!   `….experts.down_proj` `[E, H, I]`) из поштучных экспертов HF
//!   (`….experts.<e>.{gate,up,down}_proj.weight`): gate и up — строками подряд.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::error::{IoError, Result};

struct Shard {
    map: memmap2::Mmap,
    header: serde_json::Value,
    data_start: usize,
}

pub struct HfOrigin {
    dir: PathBuf,
    files: HashMap<String, String>,
    shards: Mutex<HashMap<String, Arc<Shard>>>,
}

impl std::fmt::Debug for HfOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HfOrigin({})", self.dir.display())
    }
}

fn e4m3_lut() -> &'static [f32; 256] {
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let b = i as u8;
            let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
            let e = ((b >> 3) & 0x0f) as i32;
            let m = (b & 7) as f32;
            *v = if e == 0 {
                s * m / 8.0 * 2f32.powi(-6)
            } else if e == 15 && (b & 7) == 7 {
                0.0 // NaN в весах не встречается; ноль безопаснее
            } else {
                s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
            };
        }
        t
    })
}

impl HfOrigin {
    pub fn open(dir: &Path) -> Result<Arc<Self>> {
        let idx_path = dir.join("model.safetensors.index.json");
        let files: HashMap<String, String> = if idx_path.exists() {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx_path)?)
                .map_err(|e| IoError::Bundle(format!("{}: {e}", idx_path.display())))?;
            v["weight_map"]
                .as_object()
                .ok_or_else(|| IoError::Bundle(format!("{}: нет weight_map", idx_path.display())))?
                .iter()
                .filter_map(|(k, f)| f.as_str().map(|f| (k.clone(), f.to_string())))
                .collect()
        } else {
            // Один файл без индекса.
            let single = dir.join("model.safetensors");
            let s = Self::load_shard(&single)?;
            s.header
                .as_object()
                .map(|o| o.keys().filter(|k| *k != "__metadata__").map(|k| (k.clone(), "model.safetensors".to_string())).collect())
                .unwrap_or_default()
        };
        Ok(Arc::new(Self { dir: dir.to_path_buf(), files, shards: Mutex::new(HashMap::new()) }))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn load_shard(path: &Path) -> Result<Shard> {
        let f = std::fs::File::open(path)?;
        // SAFETY: файл только читается; шарды HF не меняются во время работы.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&map[8..8 + n])
            .map_err(|e| IoError::Bundle(format!("{}: заголовок: {e}", path.display())))?;
        Ok(Shard { map, header, data_start: 8 + n })
    }

    fn shard(&self, file: &str) -> Result<Arc<Shard>> {
        let mut g = self.shards.lock().unwrap();
        if let Some(s) = g.get(file) {
            return Ok(s.clone());
        }
        let s = Arc::new(Self::load_shard(&self.dir.join(file))?);
        g.insert(file.to_string(), s.clone());
        Ok(s)
    }

    pub fn has(&self, name: &str) -> bool {
        self.files.contains_key(name)
    }

    /// Сырой тензор: (dtype, форма, байты).
    fn raw(&self, name: &str) -> Result<(String, Vec<usize>, Arc<Shard>, usize, usize)> {
        let file = self.files.get(name).ok_or_else(|| IoError::Bundle(format!("{name}: нет в оригинале")))?;
        let sh = self.shard(file)?;
        let spec = &sh.header[name];
        let dt = spec["dtype"].as_str().unwrap_or("").to_string();
        let shape: Vec<usize> = spec["shape"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()).unwrap_or_default();
        let off = spec["data_offsets"].as_array().ok_or_else(|| IoError::Bundle(format!("{name}: нет data_offsets")))?;
        let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
        let start = sh.data_start;
        Ok((dt, shape, sh, start + a, start + b))
    }

    fn floats(dt: &str, bytes: &[u8]) -> Result<Vec<f32>> {
        Ok(match dt {
            "F32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F16" => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            o => return Err(IoError::Bundle(format!("тип {o} не поддержан"))),
        })
    }

    /// Матрица HF `[n, k]` (`….weight`) в f32, FP8 — с блочными масштабами.
    pub fn matrix(&self, name: &str) -> Result<(Vec<f32>, usize, usize)> {
        let (dt, shape, sh, a, b) = self.raw(name)?;
        let [n, k] = shape.as_slice() else {
            return Err(IoError::Bundle(format!("{name}: форма {shape:?}, ожидалась матрица")));
        };
        let (n, k) = (*n, *k);
        let bytes = &sh.map[a..b];
        if dt != "F8_E4M3" {
            return Ok((Self::floats(&dt, bytes)?, n, k));
        }
        let sname = format!("{name}_scale_inv");
        let (sdt, sshape, ssh, sa, sb) = self.raw(&sname)?;
        let scales = Self::floats(&sdt, &ssh.map[sa..sb])?;
        let (bn, bk) = match sshape.as_slice() {
            [rn, rk] => (n.div_ceil(*rn), k.div_ceil(*rk)),
            _ => return Err(IoError::Bundle(format!("{sname}: форма {sshape:?}"))),
        };
        let kb = k.div_ceil(bk);
        let lut = e4m3_lut();
        let mut out = vec![0f32; n * k];
        for r in 0..n {
            let srow = &scales[(r / bn) * kb..(r / bn + 1) * kb];
            let src = &bytes[r * k..(r + 1) * k];
            let dst = &mut out[r * k..(r + 1) * k];
            for c in 0..k {
                dst[c] = lut[src[c] as usize] * srow[c / bk];
            }
        }
        Ok((out, n, k))
    }

    /// Срез `slice` тензора бандла `name` как матрица `[n, k]` из оригинала;
    /// `None` — в оригинале такого нет.
    pub fn bundle_slice(&self, name: &str, slice: usize, n: usize, k: usize) -> Option<Result<Vec<f32>>> {
        let check = |r: Result<(Vec<f32>, usize, usize)>| -> Result<Vec<f32>> {
            let (v, rn, rk) = r?;
            if (rn, rk) != (n, k) {
                return Err(IoError::Bundle(format!("{name}[{slice}]: в оригинале [{rn}, {rk}], в бандле [{n}, {k}]")));
            }
            Ok(v)
        };
        if let Some(prefix) = name.strip_suffix(".experts.gate_up_proj") {
            let g = format!("{prefix}.experts.{slice}.gate_proj.weight");
            let u = format!("{prefix}.experts.{slice}.up_proj.weight");
            if !self.has(&g) || !self.has(&u) {
                return None;
            }
            let run = || -> Result<Vec<f32>> {
                let (mut gv, gn, gk) = self.matrix(&g)?;
                let (uv, un, uk) = self.matrix(&u)?;
                if gk != uk || gn + un != n || gk != k {
                    return Err(IoError::Bundle(format!("{name}[{slice}]: gate [{gn}, {gk}] + up [{un}, {uk}] ≠ [{n}, {k}]")));
                }
                gv.extend_from_slice(&uv);
                Ok(gv)
            };
            return Some(run());
        }
        if let Some(prefix) = name.strip_suffix(".experts.down_proj") {
            let d = format!("{prefix}.experts.{slice}.down_proj.weight");
            if !self.has(&d) {
                return None;
            }
            return Some(check(self.matrix(&d)));
        }
        if slice == 0 && self.has(name) {
            return Some(check(self.matrix(name)));
        }
        None
    }
}
