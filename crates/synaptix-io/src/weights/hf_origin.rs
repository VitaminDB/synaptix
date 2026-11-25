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
    /// Префикс компонента бандла (`<prefix>.<имя HF>`): имена бандла с ним
    /// ищутся в оригинале без него.
    prefix: Option<String>,
}

/// Тензор бандла, который собирается из оригинала: имя бандла (без префикса
/// компонента), тип safetensors оригинала и форма. Поштучные эксперты HF
/// здесь уже сведены в стопки `….experts.gate_up_proj` / `….experts.down_proj`.
#[derive(Debug, Clone)]
pub struct HfTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Число поштучных экспертов, если это стопка из них (иначе `None`).
    pub stacked: Option<usize>,
}

fn dtype_size(dt: &str) -> Option<usize> {
    Some(match dt {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
        _ => return None,
    })
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&half::bf16::from_f32(*x).to_le_bytes());
    }
    out
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
        Ok(Arc::new(Self { dir: dir.to_path_buf(), files, shards: Mutex::new(HashMap::new()), prefix: None }))
    }

    /// Оригинал по списку шардов (компонент плана упаковки); `prefix` —
    /// префикс компонента в именах бандла.
    pub fn from_files(paths: &[PathBuf], prefix: Option<&str>) -> Result<Arc<Self>> {
        let dir = paths.first().and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_default();
        let mut files = HashMap::new();
        let mut shards = HashMap::new();
        for p in paths {
            let key = p.to_string_lossy().into_owned();
            let s = Self::load_shard(p)?;
            if let Some(o) = s.header.as_object() {
                for k in o.keys().filter(|k| *k != "__metadata__") {
                    if files.insert(k.clone(), key.clone()).is_some() {
                        return Err(IoError::Bundle(format!("{k}: тензор встречается в нескольких шардах")));
                    }
                }
            }
            shards.insert(key, Arc::new(s));
        }
        let prefix = prefix.filter(|p| !p.is_empty()).map(String::from);
        Ok(Arc::new(Self { dir, files, shards: Mutex::new(shards), prefix }))
    }

    /// Имя бандла → имя в оригинале (без префикса компонента).
    fn unprefix<'a>(&self, name: &'a str) -> &'a str {
        match &self.prefix {
            Some(p) => name.strip_prefix(p.as_str()).and_then(|n| n.strip_prefix('.')).unwrap_or(name),
            None => name,
        }
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
        self.files.contains_key(self.unprefix(name))
    }

    /// Тензор есть и его можно декодировать: плотный или FP8 с блочными
    /// `_scale_inv` (FP8 с одним масштабом на тензор, как таблица n-грамм
    /// Qwen3.8-Flash-Next, не берём — пусть перекодируется из бандла).
    pub fn decodable(&self, name: &str) -> bool {
        let Ok((dt, ..)) = self.raw(name) else { return false };
        match dt.as_str() {
            "BF16" | "F16" | "F32" => true,
            "F8_E4M3" => self.has(&format!("{name}_scale_inv")),
            _ => false,
        }
    }

    /// Сырой тензор: (dtype, форма, байты).
    fn raw(&self, name: &str) -> Result<(String, Vec<usize>, Arc<Shard>, usize, usize)> {
        let name = self.unprefix(name);
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

    /// Тензор как есть: (dtype safetensors, форма, байты).
    pub fn bytes(&self, name: &str) -> Result<(String, Vec<usize>, Vec<u8>)> {
        let (dt, shape, sh, a, b) = self.raw(name)?;
        Ok((dt, shape, sh.map[a..b].to_vec()))
    }

    /// Тип и форма без чтения данных.
    pub fn shape(&self, name: &str) -> Result<(String, Vec<usize>)> {
        let (dt, shape, ..) = self.raw(name)?;
        Ok((dt, shape))
    }

    /// Имена всех тензоров.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(|s| s.as_str())
    }

    fn floats(dt: &str, bytes: &[u8]) -> Result<Vec<f32>> {
        Ok(match dt {
            "F32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F16" => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            o => return Err(IoError::Bundle(format!("тип {o} не поддержан"))),
        })
    }

    /// Строки `r0..r1` двумерного плотного тензора (BF16/F16/F32) в f32 —
    /// без чтения остального: таблицы в десятки ГБ (engram Hy4) кодируются
    /// кусками. `None` — тензор не двумерный или не плотный.
    pub fn matrix_rows(&self, name: &str, r0: usize, r1: usize) -> Option<Result<Vec<f32>>> {
        let (dt, shape, sh, a, b) = self.raw(name).ok()?;
        let [n, k] = shape.as_slice() else { return None };
        let (n, k) = (*n, *k);
        let es = match dt.as_str() {
            "F32" => 4,
            "BF16" | "F16" => 2,
            _ => return None,
        };
        if r0 > r1 || r1 > n || b - a != n * k * es {
            return Some(Err(IoError::Bundle(format!("{name}: строки {r0}..{r1} при форме [{n}, {k}] и {} байт", b - a))));
        }
        Some(Self::floats(&dt, &sh.map[a + r0 * k * es..a + r1 * k * es]))
    }

    /// Матрица HF `[n, k]` (`….weight`) в f32, FP8 — с блочными масштабами.
    pub fn matrix(&self, name: &str) -> Result<(Vec<f32>, usize, usize)> {
        self.matrix_slice(name, 0)
    }

    /// Матрица `[n, k]` номер `s` из тензора HF: двумерного (`s` = 0) или
    /// стопки `[E, n, k]` (fused-эксперты); FP8 — с блочными `_scale_inv`
    /// (`[n/bn, k/bk]` или `[E, n/bn, k/bk]`).
    pub fn matrix_slice(&self, name: &str, s: usize) -> Result<(Vec<f32>, usize, usize)> {
        let (dt, shape, sh, a, b) = self.raw(name)?;
        let (e, n, k) = match shape.as_slice() {
            [n, k] => (1, *n, *k),
            [e, n, k] => (*e, *n, *k),
            _ => return Err(IoError::Bundle(format!("{name}: форма {shape:?}, ожидалась матрица или стопка"))),
        };
        if s >= e {
            return Err(IoError::Bundle(format!("{name}: среза {s} нет (их {e})")));
        }
        let es = dtype_size(&dt).ok_or_else(|| IoError::Bundle(format!("{name}: тип {dt}")))?;
        let per = n * k * es;
        if b - a != e * per {
            return Err(IoError::Bundle(format!("{name}: {} байт при форме {shape:?} {dt}", b - a)));
        }
        let bytes = &sh.map[a + s * per..a + (s + 1) * per];
        if dt != "F8_E4M3" {
            return Ok((Self::floats(&dt, bytes)?, n, k));
        }
        let sname = format!("{}_scale_inv", self.unprefix(name));
        let (sdt, sshape, ssh, sa, sb) = self.raw(&sname)?;
        let (rn, rk) = match sshape.as_slice() {
            [rn, rk] if e == 1 => (*rn, *rk),
            [se, rn, rk] if *se == e => (*rn, *rk),
            _ => return Err(IoError::Bundle(format!("{sname}: форма {sshape:?}"))),
        };
        let all = Self::floats(&sdt, &ssh.map[sa..sb])?;
        let scales = &all[s * rn * rk..(s + 1) * rn * rk];
        let (bn, bk) = (n.div_ceil(rn), k.div_ceil(rk));
        let kb = rk;
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
                return self.fused_slice(name, slice, n, k);
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
                return self.fused_slice(name, slice, n, k);
            }
            return Some(check(self.matrix(&d)));
        }
        self.fused_slice(name, slice, n, k)
    }

    /// Срез тензора, лежащего в оригинале под тем же именем (матрица или
    /// fused-стопка `[E, n, k]`).
    fn fused_slice(&self, name: &str, slice: usize, n: usize, k: usize) -> Option<Result<Vec<f32>>> {
        if !self.decodable(name) {
            return None;
        }
        let (_, shape) = self.shape(name).ok()?;
        if shape.len() == 2 && slice > 0 {
            return None;
        }
        Some(self.matrix_slice(name, slice).and_then(|(v, rn, rk)| {
            if (rn, rk) != (n, k) {
                return Err(IoError::Bundle(format!("{name}[{slice}]: в оригинале [{rn}, {rk}], в бандле [{n}, {k}]")));
            }
            Ok(v)
        }))
    }

    /// Тензоры бандла, которые даёт оригинал (имена — с префиксом компонента):
    /// `_scale_inv` при FP8-весе уходят в него, поштучные эксперты сводятся в
    /// стопки, как у `synaptix convert`. Порядок — по имени.
    pub fn bundle_tensors(&self) -> Result<Vec<HfTensor>> {
        use std::collections::{BTreeMap, BTreeSet};
        let names: BTreeSet<&str> = self.files.keys().map(|s| s.as_str()).collect();
        // Поштучные эксперты: prefix → {(i, proj)}.
        let mut experts: BTreeMap<String, BTreeSet<(usize, &'static str)>> = BTreeMap::new();
        for n in &names {
            let Some(base) = n.strip_suffix(".weight") else { continue };
            let Some((head, proj)) = base.rsplit_once('.') else { continue };
            let proj: &'static str = match proj {
                "gate_proj" => "gate_proj",
                "up_proj" => "up_proj",
                "down_proj" => "down_proj",
                _ => continue,
            };
            let Some((pre, idx)) = head.rsplit_once('.') else { continue };
            let Some(prefix) = pre.strip_suffix(".experts") else { continue };
            let Ok(i) = idx.parse::<usize>() else { continue };
            experts.entry(prefix.to_string()).or_default().insert((i, proj));
        }
        let full = |name: &str| match &self.prefix {
            Some(p) => format!("{p}.{name}"),
            None => name.to_string(),
        };
        let mut out = Vec::new();
        let mut taken: BTreeSet<String> = BTreeSet::new();
        for (prefix, set) in &experts {
            let e = set.len() / 3;
            let complete = e > 0 && set.len() == 3 * e && (0..e).all(|i| ["gate_proj", "up_proj", "down_proj"].iter().all(|p| set.contains(&(i, *p))));
            let fused = names.contains(format!("{prefix}.experts.gate_up_proj").as_str()) || names.contains(format!("{prefix}.experts.down_proj").as_str());
            if !complete || fused {
                continue;
            }
            let (gdt, gs) = self.shape(&format!("{prefix}.experts.0.gate_proj.weight"))?;
            let (_, ds) = self.shape(&format!("{prefix}.experts.0.down_proj.weight"))?;
            if gs.len() != 2 || ds.len() != 2 {
                continue;
            }
            for i in 0..e {
                for p in ["gate_proj", "up_proj", "down_proj"] {
                    let n = format!("{prefix}.experts.{i}.{p}.weight");
                    taken.insert(format!("{n}_scale_inv"));
                    taken.insert(n);
                }
            }
            out.push(HfTensor { name: full(&format!("{prefix}.experts.gate_up_proj")), dtype: gdt.clone(), shape: vec![e, 2 * gs[0], gs[1]], stacked: Some(e) });
            out.push(HfTensor { name: full(&format!("{prefix}.experts.down_proj")), dtype: gdt, shape: vec![e, ds[0], ds[1]], stacked: Some(e) });
        }
        for n in &names {
            if taken.contains(*n) {
                continue;
            }
            if let Some(base) = n.strip_suffix("_scale_inv") {
                if names.contains(base) {
                    continue;
                }
            }
            let (dt, shape) = self.shape(n)?;
            out.push(HfTensor { name: full(n), dtype: dt, shape, stacked: None });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// Тип, которым тензор ляжет в бандл без кванта: FP8 с блочными
    /// масштабами — BF16, остальное — как в оригинале.
    pub fn pass_dtype(t: &HfTensor) -> &str {
        if t.dtype == "F8_E4M3" { "BF16" } else { t.dtype.as_str() }
    }

    /// Записать тензор как есть (стопку — подряд по экспертам, FP8 — в BF16).
    pub fn write_pass(&self, t: &HfTensor, w: &mut dyn std::io::Write) -> Result<()> {
        let fp8 = t.dtype == "F8_E4M3";
        let src = |n: &str, w: &mut dyn std::io::Write| -> Result<()> {
            if fp8 {
                let (dt, shape) = self.shape(n)?;
                let slices = if shape.len() == 3 { shape[0] } else { 1 };
                if dt != "F8_E4M3" || !self.has(&format!("{}_scale_inv", self.unprefix(n))) {
                    return Err(IoError::Bundle(format!("{n}: FP8 без блочных масштабов как есть не переносится")));
                }
                for s in 0..slices {
                    let (v, ..) = self.matrix_slice(n, s)?;
                    w.write_all(&bf16_bytes(&v))?;
                }
                return Ok(());
            }
            let (_, _, sh, a, b) = self.raw(n)?;
            w.write_all(&sh.map[a..b])?;
            Ok(())
        };
        match t.stacked {
            Some(e) => {
                let base = self.unprefix(&t.name);
                if let Some(prefix) = base.strip_suffix(".experts.gate_up_proj") {
                    for i in 0..e {
                        src(&format!("{prefix}.experts.{i}.gate_proj.weight"), w)?;
                        src(&format!("{prefix}.experts.{i}.up_proj.weight"), w)?;
                    }
                } else if let Some(prefix) = base.strip_suffix(".experts.down_proj") {
                    for i in 0..e {
                        src(&format!("{prefix}.experts.{i}.down_proj.weight"), w)?;
                    }
                }
                Ok(())
            }
            None => src(&t.name, w),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};

    fn bf16(v: &[f32]) -> Vec<u8> {
        bf16_bytes(v)
    }

    /// Каталог из одного шарда: поштучные эксперты сводятся в стопки,
    /// fused-стопка режется по срезам, FP8 со стопкой масштабов `[E, nb, kb]`
    /// декодируется, `_scale_inv` в список тензоров не попадает.
    #[test]
    fn bundle_tensors_and_slices() {
        let dir = tempfile::tempdir().unwrap();
        let (n, k) = (4usize, 2usize);
        let m = |base: f32| -> Vec<f32> { (0..n * k).map(|i| base + i as f32).collect() };
        let mut data: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
        for e in 0..2 {
            for (p, b) in [("gate_proj", 0.0), ("up_proj", 100.0), ("down_proj", 200.0)] {
                data.push((format!("mtp.experts.{e}.{p}.weight"), Dtype::BF16, vec![n, k], bf16(&m(b + 10.0 * e as f32))));
            }
        }
        let fused: Vec<f32> = (0..3).flat_map(|e| m(20.0 * e as f32)).collect();
        data.push(("l.experts.gate_up_proj".into(), Dtype::BF16, vec![3, n, k], bf16(&fused)));
        // FP8: код 0x38 = 1.0; масштабы по срезу 2.0 и 4.0, блок на всю матрицу.
        data.push(("f.experts.down_proj".into(), Dtype::F8_E4M3, vec![2, n, k], vec![0x38; 2 * n * k]));
        data.push(("f.experts.down_proj_scale_inv".into(), Dtype::F32, vec![2, 1, 1], [2f32, 4.0].iter().flat_map(|x| x.to_le_bytes()).collect()));
        let views: Vec<(String, TensorView)> = data.iter().map(|(nm, dt, sh, b)| (nm.clone(), TensorView::new(*dt, sh.clone(), b).unwrap())).collect();
        let path = dir.path().join("model.safetensors");
        safetensors::serialize_to_file(views, None, &path).unwrap();

        let o = HfOrigin::from_files(&[path], None).unwrap();
        let t = o.bundle_tensors().unwrap();
        let names: Vec<&str> = t.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["f.experts.down_proj", "l.experts.gate_up_proj", "mtp.experts.down_proj", "mtp.experts.gate_up_proj"]);
        let gu = t.iter().find(|t| t.name == "mtp.experts.gate_up_proj").unwrap();
        assert_eq!((gu.shape.clone(), gu.stacked), (vec![2, 2 * n, k], Some(2)));

        // Поштучная стопка: срез 1 = gate эксперта 1, затем up.
        let s1 = o.bundle_slice("mtp.experts.gate_up_proj", 1, 2 * n, k).unwrap().unwrap();
        assert_eq!(&s1[..n * k], &m(10.0)[..]);
        assert_eq!(&s1[n * k..], &m(110.0)[..]);
        // Fused: срез 2.
        assert_eq!(o.bundle_slice("l.experts.gate_up_proj", 2, n, k).unwrap().unwrap(), m(40.0));
        // FP8-стопка: срез 1 = 1.0 × 4.0.
        assert!(o.bundle_slice("f.experts.down_proj", 1, n, k).unwrap().unwrap().iter().all(|v| *v == 4.0));
        // Перенос как есть: поштучная стопка — gate/up подряд по экспертам; FP8 — в BF16.
        let mut buf = Vec::new();
        o.write_pass(gu, &mut buf).unwrap();
        let want: Vec<u8> = [m(0.0), m(100.0), m(10.0), m(110.0)].iter().flat_map(|v| bf16(v)).collect();
        assert_eq!(buf, want);
        let f = t.iter().find(|t| t.name == "f.experts.down_proj").unwrap();
        assert_eq!(HfOrigin::pass_dtype(f), "BF16");
        let mut buf = Vec::new();
        o.write_pass(f, &mut buf).unwrap();
        let want: Vec<u8> = bf16(&[vec![2.0; n * k], vec![4.0; n * k]].concat());
        assert_eq!(buf, want);
        // С префиксом компонента имена бандла ищутся в оригинале без него.
        let p = HfOrigin::from_files(&[dir.path().join("model.safetensors")], Some("llm")).unwrap();
        assert!(p.bundle_tensors().unwrap().iter().all(|t| t.name.starts_with("llm.")));
        assert_eq!(p.bundle_slice("llm.l.experts.gate_up_proj", 1, n, k).unwrap().unwrap(), m(20.0));
    }
}
