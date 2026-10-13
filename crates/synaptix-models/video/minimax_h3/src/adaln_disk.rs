//! Дисковый кэш выходов adaLN по шагам.
//!
//! Проекции adaLN 50 блоков — 52 ГБ bf16 (две трети файла модели): их
//! чтение с диска — самая долгая фаза короткой генерации (~20 с), а результат
//! зависит только от значений шагов расписания, не от промпта и кадров. Кэш
//! хранит выходы всех блоков и финального слоя для одного шага (вектор
//! таймстепов ролей) уже в dtype кэша — повторный прогон с теми же шагами
//! берёт их с диска бит-в-бит, не читая веса. Ключ каталога — файл модели
//! (путь, размер, время изменения) и dtype; с LoRA кэш не используется.
//! `SYN_H3_ADALN_CACHE=0` — выключить; объём ограничен (старые записи
//! удаляются).

use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::loader::H3Checkpoint;

const MAGIC: &[u8; 8] = b"H3ADLN01";
/// Предел каталога кэша, байт.
const LIMIT: u64 = 6 << 30;

pub struct DiskCache {
    dir: PathBuf,
    dtype: DType,
}

fn base_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("SYNAPTIX_CACHE_DIR") {
        return Some(PathBuf::from(d));
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        return Some(PathBuf::from(d).join("synaptix"));
    }
    if let Ok(h) = std::env::var("HOME") {
        if !h.is_empty() && h != "/" {
            return Some(PathBuf::from(h).join(".cache/synaptix"));
        }
    }
    Some(PathBuf::from("/var/cache/synaptix"))
}

fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl DiskCache {
    pub fn open(ckpt: &H3Checkpoint, dtype: DType) -> Option<Self> {
        if matches!(std::env::var("SYN_H3_ADALN_CACHE").as_deref(), Ok("0")) || ckpt.has_lora() {
            return None;
        }
        let path = ckpt.source.path();
        let meta = std::fs::metadata(path).ok()?;
        let mtime = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        path.hash(&mut h);
        let key = format!(
            "{:016x}{:016x}-{:?}",
            fnv(path.as_os_str().as_encoded_bytes().iter().copied().chain(meta.len().to_le_bytes()).chain(mtime.to_le_bytes())),
            h.finish(),
            dtype
        );
        let dir = base_dir()?.join("h3-adaln").join(key);
        std::fs::create_dir_all(&dir).ok()?;
        Some(Self { dir, dtype })
    }

    fn file(&self, vals: &[f32]) -> PathBuf {
        let h = fnv(vals.iter().flat_map(|v| v.to_bits().to_le_bytes()));
        self.dir.join(format!("{h:016x}-{}.bin", vals.len()))
    }

    /// Выходы шага: `n` тензоров (блоки + финальный слой) на `device`.
    pub fn load(&self, vals: &[f32], n: usize, device: Device) -> Option<Vec<Tensor>> {
        let path = self.file(vals);
        let mut f = std::fs::File::open(&path).ok()?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).ok()?;
        let out = self.parse(&buf, vals, n, device);
        if out.is_none() {
            let _ = std::fs::remove_file(&path);
        } else {
            // Время доступа — для вытеснения старых.
            let _ = f.set_modified(std::time::SystemTime::now());
        }
        out
    }

    fn parse(&self, buf: &[u8], vals: &[f32], n: usize, device: Device) -> Option<Vec<Tensor>> {
        let mut r = Cursor { buf, pos: 0 };
        if r.take(8)? != MAGIC {
            return None;
        }
        let nv = r.u32()? as usize;
        if nv != vals.len() || (0..nv).any(|i| r.u32().map(|b| b != vals[i].to_bits()).unwrap_or(true)) {
            return None;
        }
        if r.u32()? as usize != n {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let nd = r.u32()? as usize;
            let shape: Vec<usize> = (0..nd).map(|_| r.u64().map(|d| d as usize)).collect::<Option<_>>()?;
            let len = r.u64()? as usize;
            if len != self.dtype.bytes_for_numel(shape.iter().product()) {
                return None;
            }
            let bytes = r.take(len)?;
            out.push(Tensor::from_raw_slice(bytes, shape, self.dtype, device).ok()?);
        }
        Some(out)
    }

    /// Сохранить выходы шага (тензоры в dtype кэша). Ошибки записи молча
    /// пропускаются — кэш лишь ускоряет.
    pub fn store(&self, vals: &[f32], tensors: &[Tensor]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(vals.len() as u32).to_le_bytes());
        for v in vals {
            buf.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        buf.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for t in tensors {
            if t.dtype() != self.dtype {
                return;
            }
            let Ok(bytes) = t.to_device(Device::Cpu).and_then(|c| c.contiguous()).and_then(|c| c.to_bytes()) else { return };
            buf.extend_from_slice(&(t.dims().len() as u32).to_le_bytes());
            for &d in t.dims() {
                buf.extend_from_slice(&(d as u64).to_le_bytes());
            }
            buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(&bytes);
        }
        let path = self.file(vals);
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        let ok = std::fs::File::create(&tmp).and_then(|mut f| f.write_all(&buf)).is_ok();
        if ok && std::fs::rename(&tmp, &path).is_ok() {
            prune(self.dir.parent().unwrap_or(&self.dir));
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// Удалить самые старые записи, пока кэш больше предела.
fn prune(root: &Path) {
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
    for d in std::fs::read_dir(root).into_iter().flatten().flatten() {
        for f in std::fs::read_dir(d.path()).into_iter().flatten().flatten() {
            if let Ok(m) = f.metadata() {
                files.push((m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len(), f.path()));
            }
        }
    }
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort();
    for (_, len, p) in files {
        if total <= LIMIT {
            break;
        }
        if std::fs::remove_file(&p).is_ok() {
            total -= len;
        }
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.buf.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        self.take(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn u64(&mut self) -> Option<u64> {
        self.take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }
}
